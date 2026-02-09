from audiofeat.catalog import build_feature_catalog, catalog_to_markdown, summarize_catalog


def test_feature_catalog_builds_and_renders_markdown():
    catalog = build_feature_catalog()
    summary = summarize_catalog(catalog)
    assert summary["total_components"] >= 5
    assert summary["total_features"] > 0

    markdown = catalog_to_markdown(catalog)
    assert "| Component | Function | Signature | Description |" in markdown
    assert "`audiofeat.spectral`" in markdown
