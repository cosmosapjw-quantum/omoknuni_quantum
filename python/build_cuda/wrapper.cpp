
#include <torch/extension.h>

// Forward declarations
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
    torch::Tensor leaf_values,
    torch::Tensor path_lengths,
    torch::Tensor value_sums,
    torch::Tensor visit_counts
);

PYBIND11_MODULE(custom_cuda_ops, m) {
    m.def("batched_ucb_selection", &batched_ucb_selection_cuda);
    m.def("parallel_backup", &parallel_backup_cuda);
}
