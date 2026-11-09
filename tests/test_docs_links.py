from scripts.check_docs import check_docs


def test_docs_have_no_broken_links_or_source_paths():
    problems = check_docs()
    assert not problems, "Documentation drift found:\n" + "\n".join(problems)
