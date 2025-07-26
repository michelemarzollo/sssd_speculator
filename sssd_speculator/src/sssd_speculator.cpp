/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <reader.hpp>
#include <writer.hpp>

namespace py = pybind11;

PYBIND11_MODULE(sssd_speculator, m)
{
    py::class_<Reader>(m, "Reader")
        .def(py::init<const std::string &, int, int, int, int, int, bool, int, int, int, int, int, int>(),
            py::arg("index_file_path"),
            py::arg("stop_token") = -1,
            py::arg("max_search_entries") = 100,
            py::arg("prompt_branch_length") = 8,
            py::arg("prompt_prefix_length") = 3,
            py::arg("max_output_size") = 256,
            py::arg("live_datastore") = false,
            py::arg("max_update_chunk_size") = 512 * 1024 * 1024,
            py::arg("max_indices") = 8,
            py::arg("update_interval_ms") = 20 * 60 * 1000,
            py::arg("vocab_size") = 300'000,
            py::arg("prompt_tokens_in_datastore") = 3,
            py::arg("max_topk") = 100)
        .def("get_candidates",
            &Reader::GetCandidates,
            py::arg("prefixes"),
            py::arg("decoding_lengths"),
            py::arg("branch_lengths"),
            py::arg("seq_ids"))
        .def("get_candidates_sglang",
            &Reader::GetCandidatesSglang,
            py::arg("prefixes"),
            py::arg("decoding_lengths"),
            py::arg("branch_lengths"),
            py::arg("seq_ids"))
        .def("put", &Reader::AsyncPut, py::arg("input"), py::arg("seq_id"))
        .def("sync_put", &Reader::Put, py::arg("input"), py::arg("seq_id"))
        .def("stream_put", &Reader::StreamPut, py::arg("new_tokens"), py::arg("seq_id"))
        /* new_tokens (List[Tuple[List[int], int]]): A list of tuples where each tuple contains
           new tokens and the sequence ID. */
        .def("batched_stream_put", &Reader::BatchedStreamPut, py::arg("new_tokens"))
        .def("finish_sequence", &Reader::AsyncFinishSequence, py::arg("seq_id"))
        .def("sync_finish_sequence", &Reader::FinishSequence, py::arg("seq_id"))
        .def("finish_all", &Reader::FinishAll)
        .def("update_attributes",
            &Reader::UpdateAttributes,
            py::arg("stop_token"),
            py::arg("max_search_entries"),
            py::arg("prompt_branch_length"),
            py::arg("prompt_prefix_length"),
            py::arg("max_output_size"),
            py::arg("prompt_tokens_in_datastore") = 3)
        .def("set_update_pause_flag", &Reader::SetUpdatePauseFlag, py::arg("flag"))
        .def("save_datastore", &Reader::SaveIndexesToDisk, py::arg("path"))
        .def("print_indexes", &Reader::PrintIndexes);

    py::class_<Writer>(m, "Writer")
        .def(py::init<const std::string &, int, size_t>(),
            py::arg("index_file_path"),
            py::arg("vocab_size"),
            py::arg("max_chunk_size") = 512 * 1024 * 1024)
        .def("add_entry", &Writer::AddEntry, py::arg("tokenized_sentence"))
        .def("finalize", &Writer::Finalize);
}