use rand::Rng;

/// Variable-length genome encoding all 7 criteria.
/// Only NN weights are active initially; other segments are zero-initialized
/// and will be activated as criteria are implemented.

#[derive(Clone, Debug)]
pub struct Genome {
    data: Vec<f32>,
    /// Segment layout: (start, len) for each criterion's parameters.
    /// Index 0 = NN weights, 1 = metabolic network, 2 = homeostasis params,
    /// 3 = developmental program, 4 = reproduction params, 5 = sensory params,
    /// 6 = evolution/mutation params
    segments: [(usize, usize); 7],
}

impl Genome {
    pub const METABOLIC_SIZE: usize = 16;
    pub const HOMEOSTASIS_SIZE: usize = 8;
    pub const DEVELOPMENTAL_SIZE: usize = 8;
    pub const REPRODUCTION_SIZE: usize = 4;
    pub const SENSORY_SIZE: usize = 4;
    pub const EVOLUTION_SIZE: usize = 4;

    const SEGMENT_SIZES: [usize; 6] = [
        Self::METABOLIC_SIZE,
        Self::HOMEOSTASIS_SIZE,
        Self::DEVELOPMENTAL_SIZE,
        Self::REPRODUCTION_SIZE,
        Self::SENSORY_SIZE,
        Self::EVOLUTION_SIZE,
    ];

    /// Create a genome with only NN weights active (segment 0).
    pub fn with_nn_weights(nn_weights: Vec<f32>) -> Self {
        let nn_len = nn_weights.len();
        let placeholder_sizes = Self::SEGMENT_SIZES;

        let total_len: usize = nn_len + placeholder_sizes.iter().sum::<usize>();
        let mut data = Vec::with_capacity(total_len);
        data.extend_from_slice(&nn_weights);
        // Zero-fill remaining segments
        data.resize(total_len, 0.0);

        let mut segments = [(0usize, 0usize); 7];
        segments[0] = (0, nn_len);
        let mut offset = nn_len;
        for (i, &size) in placeholder_sizes.iter().enumerate() {
            segments[i + 1] = (offset, size);
            offset += size;
        }

        Self { data, segments }
    }

    pub fn nn_weights(&self) -> &[f32] {
        self.segment_data(0)
    }

    /// Returns the parameter slice for a criterion segment (0..=6).
    pub fn segment_data(&self, criterion: usize) -> &[f32] {
        assert!(
            criterion < self.segments.len(),
            "criterion index out of range"
        );
        let (start, len) = self.segments[criterion];
        &self.data[start..start + len]
    }

    pub fn set_segment_data(&mut self, criterion: usize, data: &[f32]) {
        assert!(
            criterion < self.segments.len(),
            "criterion index out of range"
        );
        let (start, len) = self.segments[criterion];
        assert_eq!(data.len(), len, "data length must match segment size");
        self.data[start..start + len].copy_from_slice(data);
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn segments(&self) -> &[(usize, usize); 7] {
        &self.segments
    }

    /// Segment-wise crossover: for each of the 7 segments, flip a coin to pick
    /// parent_a's or parent_b's entire segment. Preserves functional linkage within
    /// segments while recombining between them.
    ///
    /// Panics if the two parents have different segment layouts.
    pub fn segment_crossover<R: Rng + ?Sized>(
        parent_a: &Genome,
        parent_b: &Genome,
        rng: &mut R,
    ) -> Genome {
        assert_eq!(
            parent_a.segments, parent_b.segments,
            "segment layouts must match for crossover"
        );
        let mut child = parent_a.clone();
        for seg_idx in 0..parent_a.segments.len() {
            if rng.random::<bool>() {
                let (start, len) = parent_b.segments[seg_idx];
                child.data[start..start + len].copy_from_slice(&parent_b.data[start..start + len]);
            }
        }
        child
    }

    /// Uniform crossover: for each gene independently, flip a coin to pick
    /// from parent_a or parent_b. This breaks segment coherence and is used
    /// as a sensitivity analysis control.
    ///
    /// Panics if the two parents have different data lengths.
    pub fn uniform_crossover<R: Rng + ?Sized>(
        parent_a: &Genome,
        parent_b: &Genome,
        rng: &mut R,
    ) -> Genome {
        assert_eq!(
            parent_a.data.len(),
            parent_b.data.len(),
            "genome lengths must match for uniform crossover"
        );
        let mut child = parent_a.clone();
        for i in 0..child.data.len() {
            if rng.random::<bool>() {
                child.data[i] = parent_b.data[i];
            }
        }
        child
    }

    pub fn mutate<R: Rng + ?Sized>(&mut self, rng: &mut R, rates: &MutationRates) {
        debug_assert!(
            rates.point_rate + rates.reset_rate + rates.scale_rate <= 1.0,
            "mutation probabilities should sum to <= 1.0"
        );
        for v in &mut self.data {
            let r = rng.random::<f32>();
            if r < rates.point_rate {
                let delta = rng.random_range(-rates.point_scale..=rates.point_scale);
                *v = (*v + delta).clamp(-rates.value_limit, rates.value_limit);
            } else if r < rates.point_rate + rates.reset_rate {
                *v = 0.0;
            } else if r < rates.point_rate + rates.reset_rate + rates.scale_rate {
                let factor = rng.random_range(rates.scale_min..=rates.scale_max);
                *v = (*v * factor).clamp(-rates.value_limit, rates.value_limit);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MutationRates {
    pub point_rate: f32,
    pub point_scale: f32,
    pub reset_rate: f32,
    pub scale_rate: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    pub value_limit: f32,
}

impl Default for MutationRates {
    fn default() -> Self {
        Self {
            point_rate: 0.02,
            point_scale: 0.15,
            reset_rate: 0.002,
            scale_rate: 0.002,
            scale_min: 0.8,
            scale_max: 1.2,
            value_limit: 2.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    #[test]
    fn mutation_is_deterministic_for_fixed_seed() {
        let mut a = Genome::with_nn_weights(vec![0.5; 16]);
        let mut b = Genome::with_nn_weights(vec![0.5; 16]);
        let mut rng_a = ChaCha12Rng::seed_from_u64(123);
        let mut rng_b = ChaCha12Rng::seed_from_u64(123);
        let rates = MutationRates::default();
        a.mutate(&mut rng_a, &rates);
        b.mutate(&mut rng_b, &rates);
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn mutation_respects_value_bounds() {
        let mut g = Genome::with_nn_weights(vec![1.5; 32]);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let rates = MutationRates::default();
        for _ in 0..100 {
            g.mutate(&mut rng, &rates);
        }
        assert!(g
            .data()
            .iter()
            .all(|v| v.is_finite() && (-rates.value_limit..=rates.value_limit).contains(v)));
    }

    #[test]
    fn segment_layout_has_correct_sizes() {
        let nn_len = 212;
        let g = Genome::with_nn_weights(vec![0.0; nn_len]);
        let segs = g.segments();

        let expected_sizes = [
            nn_len,
            Genome::METABOLIC_SIZE,
            Genome::HOMEOSTASIS_SIZE,
            Genome::DEVELOPMENTAL_SIZE,
            Genome::REPRODUCTION_SIZE,
            Genome::SENSORY_SIZE,
            Genome::EVOLUTION_SIZE,
        ];
        let mut offset = 0;
        for (i, &size) in expected_sizes.iter().enumerate() {
            assert_eq!(segs[i], (offset, size), "segment {i}");
            offset += size;
        }
        assert_eq!(g.data().len(), offset, "total genome length");
    }

    #[test]
    fn segment_data_returns_correct_slices() {
        let g = Genome::with_nn_weights(vec![1.0; 212]);
        assert_eq!(g.segment_data(1).len(), 16);
        assert!(
            g.segment_data(1).iter().all(|&v| v == 0.0),
            "non-NN segments zero-initialized"
        );
        assert_eq!(g.segment_data(4).len(), 4);
    }

    #[test]
    fn set_segment_data_overwrites_zeros() {
        let mut g = Genome::with_nn_weights(vec![0.0; 212]);
        let data = [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6,
        ];
        g.set_segment_data(1, &data);
        assert_eq!(g.segment_data(1), &data);
    }

    #[test]
    #[should_panic]
    fn set_segment_data_panics_on_wrong_size() {
        let mut g = Genome::with_nn_weights(vec![0.0; 212]);
        g.set_segment_data(1, &[0.0; 8]); // segment 1 is 16 floats
    }

    #[test]
    fn mutation_covers_all_segments() {
        let mut g = Genome::with_nn_weights(vec![0.0; 212]);
        let mut rng = ChaCha12Rng::seed_from_u64(99);
        let rates = MutationRates {
            point_rate: 0.5,
            point_scale: 1.0,
            ..MutationRates::default()
        };
        g.mutate(&mut rng, &rates);
        let non_nn_changed = g.data()[212..].iter().any(|&v| v != 0.0);
        assert!(non_nn_changed, "mutation should affect non-NN segments too");
    }

    #[test]
    fn segment_crossover_produces_mix_of_parents() {
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let parent_a = Genome::with_nn_weights(vec![1.0; 212]);
        let mut parent_b = Genome::with_nn_weights(vec![2.0; 212]);
        // Set non-NN segments differently too
        parent_b.set_segment_data(1, &[3.0; Genome::METABOLIC_SIZE]);

        let child = Genome::segment_crossover(&parent_a, &parent_b, &mut rng);
        // Child should have segments from either parent, not be identical to either
        let all_a = child.data() == parent_a.data();
        let all_b = child.data() == parent_b.data();
        // With 7 segments and fair coin, probability of all-same is (1/2)^7 ≈ 0.8%
        // Use multiple seeds to make test robust
        let mut saw_mixed = false;
        for seed in 0..10 {
            let mut r = ChaCha12Rng::seed_from_u64(seed);
            let c = Genome::segment_crossover(&parent_a, &parent_b, &mut r);
            if c.data() != parent_a.data() && c.data() != parent_b.data() {
                saw_mixed = true;
                break;
            }
        }
        assert!(saw_mixed, "crossover should mix parent segments");
    }

    #[test]
    fn segment_crossover_preserves_segment_coherence() {
        let mut rng = ChaCha12Rng::seed_from_u64(99);
        let parent_a = Genome::with_nn_weights(vec![1.0; 212]);
        let mut parent_b = Genome::with_nn_weights(vec![2.0; 212]);
        parent_b.set_segment_data(1, &[3.0; Genome::METABOLIC_SIZE]);

        let child = Genome::segment_crossover(&parent_a, &parent_b, &mut rng);
        // Each segment in child must come entirely from one parent
        for seg_idx in 0..child.segments().len() {
            let (start, len) = child.segments()[seg_idx];
            let seg = &child.data()[start..start + len];
            let seg_a = parent_a.segment_data(seg_idx);
            let seg_b = parent_b.segment_data(seg_idx);
            assert!(
                seg == seg_a || seg == seg_b,
                "segment {seg_idx} must be entirely from one parent"
            );
        }
    }

    #[test]
    fn segment_crossover_is_deterministic() {
        let parent_a = Genome::with_nn_weights(vec![1.0; 212]);
        let parent_b = Genome::with_nn_weights(vec![2.0; 212]);
        let c1 =
            Genome::segment_crossover(&parent_a, &parent_b, &mut ChaCha12Rng::seed_from_u64(7));
        let c2 =
            Genome::segment_crossover(&parent_a, &parent_b, &mut ChaCha12Rng::seed_from_u64(7));
        assert_eq!(
            c1.data(),
            c2.data(),
            "same seed should produce identical offspring"
        );
    }

    #[test]
    fn uniform_crossover_mixes_individual_genes() {
        let parent_a = Genome::with_nn_weights(vec![1.0; 212]);
        let parent_b = Genome::with_nn_weights(vec![2.0; 212]);
        let mut rng = ChaCha12Rng::seed_from_u64(42);
        let child = Genome::uniform_crossover(&parent_a, &parent_b, &mut rng);
        // Child should contain both 1.0 and 2.0 values
        let has_a = child.data().iter().any(|&v| v == 1.0);
        let has_b = child.data().iter().any(|&v| v == 2.0);
        assert!(
            has_a && has_b,
            "uniform crossover should mix individual genes"
        );
    }

    #[test]
    #[should_panic(expected = "segment layouts must match")]
    fn segment_crossover_panics_on_mismatched_layouts() {
        let parent_a = Genome::with_nn_weights(vec![1.0; 212]);
        let parent_b = Genome::with_nn_weights(vec![2.0; 100]); // different NN size
        let mut rng = ChaCha12Rng::seed_from_u64(0);
        Genome::segment_crossover(&parent_a, &parent_b, &mut rng);
    }

    proptest! {
        #[test]
        fn proptest_mutation_always_stays_within_limit(seed: u64, steps in 1usize..64) {
            let mut g = Genome::with_nn_weights(vec![0.0; 212]);
            let rates = MutationRates {
                point_rate: 0.4,
                point_scale: 2.0,
                reset_rate: 0.3,
                scale_rate: 0.2,
                scale_min: 0.2,
                scale_max: 2.5,
                value_limit: 1.5,
            };
            let mut rng = ChaCha12Rng::seed_from_u64(seed);
            for _ in 0..steps {
                g.mutate(&mut rng, &rates);
            }
            prop_assert!(g
                .data()
                .iter()
                .all(|v| v.is_finite() && (-rates.value_limit..=rates.value_limit).contains(v)));
        }
    }
}
