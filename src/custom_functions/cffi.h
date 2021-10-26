void cffi_window_edges(
    const int n, //square window size
    long long* edge_idx,
    const int num_elem, //square window size
    const long long* x,
    const long long* y,
    const long long* b,
    bool self_loop,
    long long* edges1,
    long long* edges2
);