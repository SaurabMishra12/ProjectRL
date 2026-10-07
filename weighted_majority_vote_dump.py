def weighted_majority_vote(responses: List[AgentResponse], dataset_type: str='gsm8k', W: Optional[np.ndarray]=None, agents=None, calibrator: Optional['PlattCalibrator']=None) -> str:
    """Bayesian multiplicative vote (Improvement II) with numeric normalization (I),
    spectral centrality weighting (V), stability discounting (IV), and Platt calibration (VI).

    Falls back to additive weighting gracefully when optional inputs are absent.
    """
    if not responses:
        return ''
    normalized: List[str] = [_normalize_for_vote(r.answer, dataset_type) for r in responses]
    canonical_map: Dict[str, str] = {}
    for key, r in zip(normalized, responses):
        canonical_map.setdefault(key, r.answer)
    confs = []
    for r in responses:
        c = None
        if calibrator and hasattr(calibrator, 'get_historical_accuracy'):
            c = calibrator.get_historical_accuracy(r.agent_id)
        if c is None:
            if calibrator and calibrator._fitted:
                c = calibrator.calibrate(r.confidence)
            else:
                c = r.confidence
        confs.append(max(0.51, min(0.99, c)))
    if W is not None:
        confs = spectral_vote_weights(W, confs)
        confs = [max(0.51, min(0.99, c)) for c in confs]
    if agents is not None:
        stability = compute_stability_weights(agents, len(responses))
        confs = [c * stability.get(r.agent_id, 1.0) for c, r in zip(confs, responses)]
        confs = [max(0.51, min(0.99, c)) for c in confs]
    unique_answers = list(set(normalized))
    if len(unique_answers) == 1:
        return canonical_map[unique_answers[0]]
    n_options = max(len(unique_answers), 2)
    log_posterior: Dict[str, float] = {}
    for candidate in unique_answers:
        log_p = 0.0
        for ans, theta in zip(normalized, confs):
            if ans == candidate:
                log_p += np.log(theta)
            else:
                log_p += np.log((1.0 - theta) / (n_options - 1))
        log_posterior[candidate] = log_p
    best_key = max(log_posterior, key=log_posterior.get)
    if dataset_type in ['gsm8k', 'aqua']:
        _num_entries = []
        for _key, _r, _c in zip(normalized, responses, confs):
            _nums = re.findall('-?\\d+\\.?\\d*', _key)
            if _nums:
                try:
                    _num_entries.append((float(_nums[-1]), _r.agent_id, canonical_map[_key]))
                except (ValueError, IndexError):
                    pass
        if len(_num_entries) == len(responses) and len(_num_entries) > 0:
            if calibrator and hasattr(calibrator, 'get_variance'):
                _vars = np.array([calibrator.get_variance(aid) for _, aid, _ in _num_entries])
            else:
                _vars = np.ones(len(_num_entries))
            _precisions = 1.0 / (_vars + 0.0001)
            _prec_by_val: Dict[str, float] = {}
            _repr_by_val: Dict[str, str] = {}
            for (_v, _aid, _orig), _prec in zip(_num_entries, _precisions):
                _vkey = str(int(_v)) if float(_v).is_integer() else str(round(_v, 4))
                _prec_by_val[_vkey] = _prec_by_val.get(_vkey, 0.0) + _prec
                _repr_by_val.setdefault(_vkey, _orig)
            _best_vkey = max(_prec_by_val, key=_prec_by_val.__getitem__)
            return _repr_by_val[_best_vkey]
    return canonical_map[best_key]
