//! IR-level smoke test for the RLX NeuroRVQ FM branch graph.

use neurorvq_rs::rlx::graph::{build_fm_branch_graph, FmBranchSpec};

fn zero_fill_params(graph: &rlx::Graph, compiled: &mut rlx::CompiledGraph) {
    use rlx::Op;
    for node in graph.nodes() {
        let Op::Param { name } = &node.op else {
            continue;
        };
        let n = node
            .shape
            .num_elements()
            .expect("param shape must be static");
        compiled.set_param(name, &vec![0.0; n]);
    }
}

#[test]
fn fm_branch_graph_compiles_and_runs() {
    let spec = FmBranchSpec {
        b: 1,
        s: 9,
        seq_len: 8,
        d: 16,
        out_dim: 16,
        nh: 2,
        dh: 8,
        depth: 2,
        ff: 64,
        norm_eps: 1e-6,
        block_prefix: "blocks".into(),
        head_prefix: String::new(),
        branch: 1,
        use_qk_norm: true,
    };

    let graph = build_fm_branch_graph(&spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());
    zero_fill_params(&graph, &mut compiled);

    let x = vec![0.1_f32; spec.b * spec.s * spec.d];
    let outs = compiled.run(&[("x", &x)]);
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].len(), spec.b * spec.seq_len * spec.out_dim);
}

#[test]
fn encoder_branch_graph_compiles_and_runs() {
    let spec = FmBranchSpec {
        b: 1,
        s: 5,
        seq_len: 4,
        d: 16,
        out_dim: 16,
        nh: 2,
        dh: 8,
        depth: 2,
        ff: 64,
        norm_eps: 1e-6,
        block_prefix: "encoder.blocks".into(),
        head_prefix: "encoder".into(),
        branch: 2,
        use_qk_norm: true,
    };

    let graph = build_fm_branch_graph(&spec);
    let mut compiled = rlx::Session::new(rlx::Device::Cpu).compile(graph.clone());
    zero_fill_params(&graph, &mut compiled);

    let x = vec![0.0_f32; spec.b * spec.s * spec.d];
    let outs = compiled.run(&[("x", &x)]);
    assert_eq!(outs[0].len(), spec.b * spec.seq_len * spec.out_dim);
}
