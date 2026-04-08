def test_hypothesis_monitor_becomes_more_suspicious_under_input_flip_fault():
    """Injected control-sign faults should increase the e-process."""
    import torch
    from monitor import HypothesisTestingMonitor
    from monitor.adapters.neural_clbf_pendulum import NeuralCLBFPendulum

    initial_state = torch.tensor([1.0, 0.5])
    max_steps = 400

    nominal = NeuralCLBFPendulum(
        dt=0.01,
        noise_level=0.0,
        flip_inputs_prob_to=0.0,
        flip_inputs_prob_from=0.0,
    )
    faulty = NeuralCLBFPendulum(
        dt=0.01,
        noise_level=0.0,
        flip_inputs_prob_to=1.0,
        flip_inputs_prob_from=0.0,
    )

    nominal.reset(initial_state=initial_state)
    faulty.reset(initial_state=initial_state)

    nominal_monitor = HypothesisTestingMonitor(delta=0.01)
    faulty_monitor = HypothesisTestingMonitor(delta=0.01)

    nominal_max_e = 1.0
    faulty_max_e = 1.0
    faulty_rejected = False

    for step, ((nominal_verdict, nominal_info), (faulty_verdict, faulty_info)) in enumerate(
        zip(nominal_monitor(nominal), faulty_monitor(faulty)),
        start=1,
    ):
        nominal_max_e = max(nominal_max_e, float(nominal_info["e_value"]))
        faulty_max_e = max(faulty_max_e, float(faulty_info["e_value"]))
        if faulty_verdict == "F":
            faulty_rejected = True
            break
        if step >= max_steps:
            break

    assert faulty_max_e > nominal_max_e, (
        f"Expected input-flip fault to raise suspicion, got nominal_max_e={nominal_max_e}, "
        f"faulty_max_e={faulty_max_e}"
    )
    assert faulty_rejected or faulty_max_e >= 2.0, (
        f"Expected strong suspicion under flip fault, got faulty_max_e={faulty_max_e}"
    )
