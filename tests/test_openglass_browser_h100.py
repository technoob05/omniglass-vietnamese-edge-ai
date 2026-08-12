from scripts.openglass_browser_h100 import concise_answer, build_app


def test_concise_answer_caps_realtime_speech():
    value = concise_answer(" ".join(f"từ{i}" for i in range(30)))
    assert len(value.split()) == 18
    assert value.endswith(".")


def test_browser_adapter_has_same_origin_routes_and_local_template():
    app = build_app("http://127.0.0.1:8780/")
    routes = {(route.method, route.resource.canonical) for route in app.router.routes()}
    assert ("GET", "/") in routes
    assert ("GET", "/api/health") in routes
    assert ("POST", "/api/turn") in routes
    assert app["agent_url"] == "http://127.0.0.1:8780"
    assert app["index_path"].is_file()
