clean:
	rm -rvf docs/src/_autogen docs/src/_autosummary
	find -type d -name "*__pycache__" -exec rm -rv {} \;

autodoc:
	./docs/autorebuild.sh

test:
	pytest $(ARGS)
