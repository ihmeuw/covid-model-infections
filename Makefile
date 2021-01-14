# makefile for easy manage package
.PHONY: clean

clean:
	find . -name "*.so*" | xargs rm -rf
	find . -name "*.pyc" | xargs rm -rf
	find . -name "__pycache__" | xargs rm -rf
	find . -name "build" | xargs rm -rf
	find . -name "dist" | xargs rm -rf
	find . -name "MANIFEST" | xargs rm -rf
	find . -name "*.egg-info" | xargs rm -rf
	find . -name ".pytest_cache" | xargs rm -rf
	rm -rf limetr MRTool

install_env:
	( \
        git clone git@github.com:zhengp0/limetr.git && \
        git clone git@github.com:ihmeuw-msca/MRTool.git && \
        eval "$($(CONDA_PREFIX)/conda shell.bash hook)" && \
		conda create -n $(ENV_NAME) -y -c conda-forge cyipopt python=3.7 && \
		source $(CONDA_PREFIX)/activate $(ENV_NAME) && \
		conda install --yes h5py && \
		pip install --extra-index-url https://artifactory.ihme.washington.edu/artifactory/api/pypi/pypi-shared/simple/ \
                  numpy scipy pandas matplotlib seaborn pyyaml dill loguru pytest xspline && \
		cd limetr && make install && cd .. && \
		cd MRTool && python setup.py install && cd .. && \
		pip install -e . ; \
    )

.PHONY: test
test:
	pytest -vv tests


uninstall_env:
	conda remove --name $(ENV_NAME) --all
