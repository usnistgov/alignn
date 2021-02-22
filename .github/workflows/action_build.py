name: JARVIS-DGL github action

on: [push, pull_request]
jobs:
  miniconda:
    name: Miniconda ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
        matrix:
            os: ["ubuntu-latest"]
    steps:
      - uses: actions/checkout@v2
      - uses: conda-incubator/setup-miniconda@v2
       
        with:
          activate-environment: test
        
          python-version: 3.8
          auto-activate-base: false
      - shell: bash -l {0}
        run: |
          conda info
          conda list
     
      - name: Run pytest
        shell: bash -l {0}
        run: |
            python setup.py develop
            pip install codecov pytest pytest-cov coverage
            coverage run -m pytest
            coverage report -m
            codecov

