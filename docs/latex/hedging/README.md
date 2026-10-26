# Learning to hedge with stocks and options

The canonical manuscript is [volt_jormungandr_option_hedging.tex](volt_jormungandr_option_hedging.tex);
[the PDF](volt_jormungandr_option_hedging.pdf) is built from that source.
It develops the market model, observations, holdings, financed account, terminal
risk objective and direct policy optimization, then shows the actual training,
evaluation, inference and checkpoint calls through Jormungandr and Volt.

Build from the repository root, with TeX Live and latexmk installed:

```sh
python docs/latex/hedging/build.py
```

Build intermediates go in the local ignored `build/` directory. The script checks
LaTeX layout and references, then places the finished PDF beside its source.
This directory carries its own journal class and 8pt style so the other
Jormungandr manuscript keeps its existing template.

The complete [client example](../../../examples/native_hedging_api.py) creates a
learner, uploads separate training and validation batches, trains, evaluates,
infers holdings and restores a checkpoint. Its module documentation gives the
service launch commands. The [API guide](../../native-learners.md) describes
the endpoint schemas and supported policy configuration.

Both the original two-actor example and this client use synthetic market paths.
