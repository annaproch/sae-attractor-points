# Third-party notices

These notices identify third-party material included in this repository and its
experiment outputs. They do not grant permissions beyond the upstream terms or
assign a license to this project's original code. Source information was checked
on 2026-11-04.

## Wikipedia text distributed through WikiText

Credit: Wikipedia contributors, and the WikiText dataset authors Stephen Merity,
Caiming Xiong, James Bradbury, and Richard Socher, associated with Salesforce.
Dataset paper: [Pointer Sentinel Mixture Models (2016)](https://arxiv.org/abs/1609.07843).

Source: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext),
configuration `wikitext-2-raw-v1`, `test` split. The sample and its provenance are
in `data/natural-text-001/texts.jsonl` and `data/natural-text-001/manifest.json`.
The same text is embedded in `runs/natural-001/probes.json`; this notice also
applies to copies and excerpts of that source text in other experiment outputs.

The upstream dataset card's metadata lists `cc-by-sa-3.0` and `gfdl`, while its
"Licensing Information" section specifies CC BY-SA 4.0. This notice records that
discrepancy without asserting that the upstream material has been relicensed.
Relevant license texts are:

- [Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/legalcode)
- [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/legalcode)
- [GNU Free Documentation License versions](https://www.gnu.org/licenses/old-licenses/fdl.html)

The Wikipedia-derived text remains subject to its applicable upstream license.
Preserve appropriate attribution and license information when redistributing it,
and follow the applicable ShareAlike terms for adaptations. The upstream card
does not specify a GFDL version in its metadata.

Local processing: `scripts/prepare_text.py` scans the first 1,000 source rows,
filters for paragraphs of at least 80 words, samples 100 with seed 42, strips
leading/trailing whitespace, and serializes the paragraphs as JSONL with IDs.
WikiText's existing text preprocessing is retained. Experiment records add
tokenization and numerical measurements. Each paragraph's `source_row`, together
with the configuration and split above, identifies its row in the source dataset;
the local sample does not include individual Wikipedia article revision IDs.

## Apollo Research roleplaying data and grading template

Credit: Apollo Research and contributors to
[deception-detection](https://github.com/ApolloResearch/deception-detection),
the repository for *Detecting Strategic Deception Using Linear Probes*.

Upstream sources:

- [Roleplaying dataset](https://github.com/ApolloResearch/deception-detection/blob/main/data/roleplaying/dataset.yaml)
- [Roleplaying grading template](https://github.com/ApolloResearch/deception-detection/blob/main/data/roleplaying/grading_template.txt)

Bundled material includes `runs/deception-pilot-001/source.yaml`,
`runs/deception-pilot-002/source.yaml`, `runs/deception-smoke-001/source.yaml`,
and `data/deception/apollo_grading_template.txt`. Selected scenarios and prompt
text also occur in the corresponding `prompts.json` and `responses.jsonl` files,
`runs/deception-activations-001/examples.json`, and deception trajectory/sweep
probe files and related records.

The pilot saves the downloaded dataset bytes and records their SHA-256 in run
manifests. It selects scenarios and combines their fields into a Gemma user
message; generated responses, labels, and trajectory measurements are local
experiment outputs. Embedded upstream text retains its original provenance.

No explicit redistribution license was identified in the upstream repository
root, README, or package metadata during review. Its public disclosure status
does not itself establish a redistribution license. This attribution notice
does not resolve that permission question or claim an open-source license for
the copied material.

## Adapted activation-steering prompts

Credit: Jonathan Bostock,
[activation-steering-sweep](https://github.com/jonathanbostock/activation-steering-sweep),
specifically the caracara contrastive statements in
[`steering_sweep.py`](https://github.com/jonathanbostock/activation-steering-sweep/blob/main/steering_sweep.py).

The eight owl/hawk statement pairs are represented as parameterized templates in
`scripts/steering_pilot.py` and saved in
`runs/steering-pilot-001/contrastive_prompts.json`. Their experimental adaptation
is described in `STEERING_PILOT.md`: this project uses layer 17 and a matching SAE
rather than the source experiment's layer 25.

No explicit license was identified on the upstream repository page during
review. This notice provides attribution and makes no additional permission
claim for the adapted source material.
