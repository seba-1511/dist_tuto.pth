
PHONY: all

all:
	paperify.py tuto.md paper & paperify.py tuto.md web
	cp tuto.html index.html

ptp:
	python ptp.py
