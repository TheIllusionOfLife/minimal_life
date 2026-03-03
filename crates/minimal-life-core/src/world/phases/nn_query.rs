use crate::spatial;
use crate::spatial::AgentLocation;
use rayon::prelude::*;
use rstar::RTree;

use super::super::World;

/// Per-agent result computed in parallel: (delta, organism_index, neighbor_count).
type AgentResult = ([f32; 4], usize, usize);

impl World {
    /// Compute neighbor-informed neural deltas for all agents.
    ///
    /// When `config.parallel_nn` is true, spatial queries and NN forward passes
    /// run in parallel via Rayon. The organism-level neighbor accumulation is
    /// merged sequentially afterwards (cheap O(agents) pass).
    pub(in crate::world) fn step_nn_query_phase(&mut self, tree: &RTree<AgentLocation>) {
        let neighbor_sums = &mut self.neighbor_sums_buffer;
        let neighbor_counts = &mut self.neighbor_counts_buffer;
        let agents = &self.agents;
        let organisms = &self.organisms;
        let config = &self.config;

        let org_count = organisms.len();
        if neighbor_sums.len() != org_count {
            neighbor_sums.resize(org_count, 0.0);
            neighbor_counts.resize(org_count, 0);
        }
        neighbor_sums.fill(0.0);
        neighbor_counts.fill(0);

        let compute_agent = |agent: &super::super::Agent| -> AgentResult {
            let org_idx = agent.organism_id as usize;
            if !organisms.get(org_idx).map(|o| o.alive).unwrap_or(false) {
                return ([0.0; 4], org_idx, 0);
            }

            let dev_sensing = if config.enable_growth {
                organisms[org_idx]
                    .developmental_program
                    .stage_factors(organisms[org_idx].maturity)
                    .1
            } else {
                1.0
            };
            let effective_radius = config.sensing_radius * dev_sensing as f64;

            let neighbor_count = spatial::count_neighbors_topo(
                tree,
                agent.position,
                effective_radius,
                agent.id,
                config.world_size,
                config.world_topology,
            );

            let input: [f32; 8] = [
                (agent.position[0] / config.world_size) as f32,
                (agent.position[1] / config.world_size) as f32,
                (agent.velocity[0] / config.max_speed) as f32,
                (agent.velocity[1] / config.max_speed) as f32,
                agent.internal_state[0],
                agent.internal_state[1],
                agent.internal_state[2],
                neighbor_count as f32 / config.neighbor_norm as f32,
            ];
            let nn = &organisms[org_idx].nn;
            (nn.forward(&input), org_idx, neighbor_count)
        };

        // Compute all agent results (parallel or sequential based on config)
        let results: Vec<AgentResult> = if config.parallel_nn {
            agents.par_iter().map(compute_agent).collect()
        } else {
            agents.iter().map(compute_agent).collect()
        };

        // Sequential merge: fill deltas buffer and accumulate neighbor stats
        self.deltas_buffer.clear();
        self.deltas_buffer.reserve(results.len());
        for (delta, org_idx, nc) in results {
            self.deltas_buffer.push(delta);
            neighbor_sums[org_idx] += nc as f32;
            neighbor_counts[org_idx] += 1;
        }
    }
}
