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
        source $(CONDA_PREFIX)/etc/profile.d/conda.sh && \
		conda create -n $(ENV_NAME) -y -c conda-forge python=3.7 cyipopt gmp && \
		conda activate $(ENV_NAME) && \
		conda install --yes h5py && \
		pip install --global-option=build_ext --global-option '-I$(CONDA_PREFIX)/envs/$(ENV_NAME)/include/' pycddlib && \
		cd limetr && make install && cd .. && \
		cd MRTool && python setup.py install && cd .. && \
		pip install -e . ; \
	)

.PHONY: test
test:
	pytest -vv tests


uninstall_env:
	( \
		source $(CONDA_PREFIX)/etc/profile.d/conda.sh && \
		conda remove --name $(ENV_NAME) --all ; \
	)
