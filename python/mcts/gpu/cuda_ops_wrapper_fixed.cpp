// C++ wrapper for CUDA operations with proper torch operator registration
#include <torch/extension.h>

// Declare the CUDA functions with their actual signatures
torch::Tensor batched_ucb_selection_cuda(
    torch::Tensor q_values,
    torch::Tensor visit_counts, 
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    float c_puct
);

torch::Tensor parallel_backup_cuda(
    torch::Tensor paths,
    torch::Tensor values,
    torch::Tensor path_lengths,
    torch::Tensor value_sums,
    torch::Tensor visit_counts
);

torch::Tensor batched_add_children_cuda(
    torch::Tensor parent_indices,
    torch::Tensor actions,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor edge_visits,
    torch::Tensor edge_values,
    torch::Tensor edge_priors,
    torch::Tensor child_indices,
    torch::Tensor parent_actions,
    torch::Tensor node_priors,
    torch::Tensor visit_counts,
    torch::Tensor value_sums,
    torch::Tensor num_children,
    torch::Tensor edge_count,
    int batch_size,
    int max_children
);

torch::Tensor quantum_interference_cuda(
    torch::Tensor amplitudes,
    torch::Tensor phases,
    torch::Tensor similarities,
    torch::Tensor path_lengths,
    float alpha,
    float beta,
    float gamma
);

torch::Tensor evaluate_gomoku_positions_cuda(
    torch::Tensor boards,
    torch::Tensor current_players
);

// Wrapper functions that convert types
torch::Tensor batched_ucb_selection_wrapper(
    torch::Tensor q_values,
    torch::Tensor visit_counts, 
    torch::Tensor parent_visits,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    double c_puct
) {
    return batched_ucb_selection_cuda(
        q_values, visit_counts, parent_visits, priors, 
        row_ptr, col_indices, static_cast<float>(c_puct)
    );
}

torch::Tensor batched_add_children_wrapper(
    torch::Tensor parent_indices,
    torch::Tensor actions,
    torch::Tensor priors,
    torch::Tensor row_ptr,
    torch::Tensor col_indices,
    torch::Tensor edge_visits,
    torch::Tensor edge_values,
    torch::Tensor edge_priors,
    torch::Tensor child_indices,
    torch::Tensor parent_actions,
    torch::Tensor node_priors,
    torch::Tensor visit_counts,
    torch::Tensor value_sums,
    torch::Tensor num_children,
    torch::Tensor edge_count,
    int64_t batch_size,
    int64_t max_children
) {
    return batched_add_children_cuda(
        parent_indices, actions, priors, row_ptr, col_indices,
        edge_visits, edge_values, edge_priors, child_indices,
        parent_actions, node_priors, visit_counts, value_sums,
        num_children, edge_count, 
        static_cast<int>(batch_size), static_cast<int>(max_children)
    );
}

torch::Tensor quantum_interference_wrapper(
    torch::Tensor amplitudes,
    torch::Tensor phases,
    torch::Tensor similarities,
    torch::Tensor path_lengths,
    double alpha,
    double beta,
    double gamma
) {
    return quantum_interference_cuda(
        amplitudes, phases, similarities, path_lengths,
        static_cast<float>(alpha), static_cast<float>(beta), static_cast<float>(gamma)
    );
}

// Register as torch custom operators
TORCH_LIBRARY(custom_cuda_ops, m) {
    m.def("batched_ucb_selection(Tensor q_values, Tensor visit_counts, Tensor parent_visits, Tensor priors, Tensor row_ptr, Tensor col_indices, float c_puct) -> Tensor");
    m.def("parallel_backup(Tensor paths, Tensor values, Tensor path_lengths, Tensor value_sums, Tensor visit_counts) -> Tensor");
    m.def("batched_add_children(Tensor parent_indices, Tensor actions, Tensor priors, Tensor row_ptr, Tensor col_indices, Tensor edge_visits, Tensor edge_values, Tensor edge_priors, Tensor child_indices, Tensor parent_actions, Tensor node_priors, Tensor visit_counts, Tensor value_sums, Tensor num_children, Tensor edge_count, int batch_size, int max_children) -> Tensor");
    m.def("quantum_interference(Tensor amplitudes, Tensor phases, Tensor similarities, Tensor path_lengths, float alpha, float beta, float gamma) -> Tensor");
    m.def("evaluate_gomoku_positions(Tensor boards, Tensor current_players) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_cuda_ops, CUDA, m) {
    m.impl("batched_ucb_selection", batched_ucb_selection_wrapper);
    m.impl("parallel_backup", parallel_backup_cuda);
    m.impl("batched_add_children", batched_add_children_wrapper);
    m.impl("quantum_interference", quantum_interference_wrapper);
    m.impl("evaluate_gomoku_positions", evaluate_gomoku_positions_cuda);
}