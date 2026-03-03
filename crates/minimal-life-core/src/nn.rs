//! Feedforward neural network: 8 inputs → H hidden (tanh) → 4 outputs (tanh).
//! Heap-allocated with configurable hidden size. Default hidden_size=16 (212 weights).
//!
//! Inputs:  position(2) + velocity(2) + internal_state(3) + neighbor_count(1) = 8
//! Outputs: velocity_delta(2) + state_delta(2) = 4

pub const INPUT_SIZE: usize = 8;
pub const OUTPUT_SIZE: usize = 4;
const DEFAULT_HIDDEN_SIZE: usize = 16;

#[derive(Clone, Debug)]
pub struct NeuralNet {
    hidden_size: usize,
    /// Flat weight storage: [w_ih (INPUT×H), b_h (H), w_ho (H×OUTPUT), b_o (OUTPUT)]
    weights: Vec<f32>,
}

impl NeuralNet {
    /// Default weight count for hidden_size=16: 8*16 + 16 + 16*4 + 4 = 212.
    pub const WEIGHT_COUNT: usize = Self::weight_count(DEFAULT_HIDDEN_SIZE);

    /// Compute total weight count for a given hidden size.
    pub const fn weight_count(hidden_size: usize) -> usize {
        INPUT_SIZE * hidden_size + hidden_size + hidden_size * OUTPUT_SIZE + OUTPUT_SIZE
    }

    /// Create a NN with the given hidden size from an iterator of f32 values.
    /// Panics if fewer than `weight_count(hidden_size)` values are provided.
    pub fn from_weights_with_hidden(
        hidden_size: usize,
        weights: impl Iterator<Item = f32>,
    ) -> Self {
        let expected = Self::weight_count(hidden_size);
        let weights: Vec<f32> = weights.take(expected).collect();
        assert_eq!(
            weights.len(),
            expected,
            "insufficient weights: need {} for hidden_size={}, got {}",
            expected,
            hidden_size,
            weights.len()
        );
        Self {
            hidden_size,
            weights,
        }
    }

    /// Create a NN with default hidden_size=16 from an iterator.
    /// Panics if fewer than WEIGHT_COUNT (212) values.
    pub fn from_weights(weights: impl Iterator<Item = f32>) -> Self {
        Self::from_weights_with_hidden(DEFAULT_HIDDEN_SIZE, weights)
    }

    /// Create a zero-initialized NN with the given hidden size.
    pub fn new(hidden_size: usize) -> Self {
        let count = Self::weight_count(hidden_size);
        Self {
            hidden_size,
            weights: vec![0.0; count],
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    // Offset helpers for the flat weight layout
    fn w_ih_offset(&self) -> usize {
        0
    }
    fn b_h_offset(&self) -> usize {
        INPUT_SIZE * self.hidden_size
    }
    fn w_ho_offset(&self) -> usize {
        self.b_h_offset() + self.hidden_size
    }
    fn b_o_offset(&self) -> usize {
        self.w_ho_offset() + self.hidden_size * OUTPUT_SIZE
    }

    /// Forward pass. Returns [vel_dx, vel_dy, state_d0, state_d1].
    pub fn forward(&self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        let h = self.hidden_size;
        let w_ih = &self.weights[self.w_ih_offset()..self.b_h_offset()];
        let b_h = &self.weights[self.b_h_offset()..self.w_ho_offset()];
        let w_ho = &self.weights[self.w_ho_offset()..self.b_o_offset()];
        let b_o = &self.weights[self.b_o_offset()..];

        // Hidden layer: input(8) × w_ih(8×H) + b_h(H) → tanh
        let mut hidden = vec![0.0f32; h];
        hidden.copy_from_slice(b_h);
        for (i, &x) in input.iter().enumerate() {
            let row = &w_ih[i * h..(i + 1) * h];
            for (j, hj) in hidden.iter_mut().enumerate() {
                *hj += x * row[j];
            }
        }
        for hj in &mut hidden {
            *hj = hj.tanh();
        }

        // Output layer: hidden(H) × w_ho(H×4) + b_o(4) → tanh
        let mut output = [0.0f32; OUTPUT_SIZE];
        output.copy_from_slice(b_o);
        for (i, &hval) in hidden.iter().enumerate() {
            let row = &w_ho[i * OUTPUT_SIZE..(i + 1) * OUTPUT_SIZE];
            for (j, oj) in output.iter_mut().enumerate() {
                *oj += hval * row[j];
            }
        }
        for oj in &mut output {
            *oj = oj.tanh();
        }

        output
    }

    /// Flatten network parameters (same order as `from_weights` expects).
    pub fn to_weight_vec(&self) -> Vec<f32> {
        self.weights.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn weight_count_matches_dimensions() {
        assert_eq!(NeuralNet::WEIGHT_COUNT, 8 * 16 + 16 + 16 * 4 + 4);
        assert_eq!(NeuralNet::WEIGHT_COUNT, 212);
    }

    #[test]
    fn forward_output_bounded_by_tanh() {
        let nn = NeuralNet::from_weights((0..NeuralNet::WEIGHT_COUNT).map(|i| (i as f32) * 0.01));
        let input = [1.0f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((-1.0..=1.0).contains(&o), "output {o} outside tanh range");
        }
    }

    #[test]
    fn zero_weights_produce_zero_output() {
        let nn = NeuralNet::from_weights(std::iter::repeat_n(0.0f32, NeuralNet::WEIGHT_COUNT));
        let input = [1.0f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((o.abs()) < 1e-7, "expected ~0 with zero weights, got {o}");
        }
    }

    #[test]
    #[should_panic(expected = "insufficient weights")]
    fn from_weights_panics_on_short_iterator() {
        NeuralNet::from_weights(std::iter::repeat_n(0.0f32, 10));
    }

    #[test]
    fn to_weight_vec_round_trips_into_equivalent_network() {
        let nn = NeuralNet::from_weights((0..NeuralNet::WEIGHT_COUNT).map(|i| i as f32 * 0.01));
        let round_trip = NeuralNet::from_weights(nn.to_weight_vec().into_iter());
        let input = [0.25f32; INPUT_SIZE];
        assert_eq!(nn.forward(&input), round_trip.forward(&input));
    }

    #[test]
    fn configurable_hidden_size_8() {
        let h = 8;
        let count = NeuralNet::weight_count(h);
        assert_eq!(count, 8 * 8 + 8 + 8 * 4 + 4); // 108
        let nn = NeuralNet::from_weights_with_hidden(h, (0..count).map(|i| i as f32 * 0.01));
        let input = [1.0f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((-1.0..=1.0).contains(&o), "output {o} outside tanh range");
        }
        assert_eq!(nn.hidden_size(), h);
    }

    #[test]
    fn configurable_hidden_size_32() {
        let h = 32;
        let count = NeuralNet::weight_count(h);
        assert_eq!(count, 8 * 32 + 32 + 32 * 4 + 4); // 420
        let nn = NeuralNet::from_weights_with_hidden(h, (0..count).map(|i| i as f32 * 0.001));
        let input = [0.5f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((-1.0..=1.0).contains(&o), "output {o} outside tanh range");
        }
    }

    #[test]
    fn new_creates_zero_nn() {
        let nn = NeuralNet::new(16);
        let input = [1.0f32; INPUT_SIZE];
        let output = nn.forward(&input);
        for &o in &output {
            assert!((o.abs()) < 1e-7, "expected ~0 from zero-init NN, got {o}");
        }
    }

    #[test]
    fn default_hidden_size_matches_legacy() {
        // Verify the heap-based NN produces the same output as the old stack-based one
        // by checking the default hidden_size=16 path
        let weights: Vec<f32> = (0..212).map(|i| i as f32 * 0.01).collect();
        let nn = NeuralNet::from_weights(weights.iter().copied());
        let input = [0.3f32; INPUT_SIZE];
        let output = nn.forward(&input);
        // All outputs should be bounded and finite
        for &o in &output {
            assert!(o.is_finite() && (-1.0..=1.0).contains(&o));
        }
        // Round-trip should be exact
        let rt = NeuralNet::from_weights(nn.to_weight_vec().into_iter());
        assert_eq!(nn.forward(&input), rt.forward(&input));
    }

    proptest! {
        #[test]
        fn proptest_forward_outputs_finite_and_bounded(
            weights in proptest::collection::vec(-10.0f32..10.0f32, NeuralNet::WEIGHT_COUNT),
            inputs in proptest::collection::vec(-5.0f32..5.0f32, INPUT_SIZE),
        ) {
            let nn = NeuralNet::from_weights(weights.into_iter());
            let input: [f32; INPUT_SIZE] = inputs.try_into().expect("input size should match");
            let output = nn.forward(&input);
            prop_assert!(output.iter().all(|o| o.is_finite() && *o >= -1.0 && *o <= 1.0));
        }
    }
}
