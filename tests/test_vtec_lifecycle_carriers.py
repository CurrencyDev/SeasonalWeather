from seasonalweather.alerts.vtec import same_codes_for_vtec, toneout_policy


def test_warning_continuation_vtec_exposes_underlying_same_code():
    vtec = ["/O.CON.KLWX.SV.W.0050.260519T2104Z-260519T2130Z/"]

    policy = toneout_policy(vtec)

    assert policy.mode == "VOICE"
    assert policy.same_code is None
    assert same_codes_for_vtec(vtec) == ["SVR"]


def test_warning_expiration_vtec_exposes_underlying_same_code():
    vtec = ["/O.EXP.KLWX.SV.W.0050.260519T2104Z-260519T2130Z/"]

    policy = toneout_policy(vtec)

    assert policy.mode == "VOICE"
    assert policy.same_code is None
    assert same_codes_for_vtec(vtec) == ["SVR"]


def test_marine_statement_carrier_vtec_exposes_special_marine_warning_code():
    vtec = ["/O.CON.KLWX.MA.W.0057.260519T2131Z-260519T2300Z/"]

    assert toneout_policy(vtec).mode == "VOICE"
    assert same_codes_for_vtec(vtec) == ["SMW"]
