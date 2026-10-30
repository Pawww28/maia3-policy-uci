# maia3-policy-uci

[Maia 3](https://github.com/CSSLab/maia3) (CSSLab, University of Toronto) as a UCI
engine that also says how sure the model was of the move it played.

[Chess with Me](https://chess-with-me.com) uses it when you play Mira at a rating:
the probability decides how long Mira "thinks" before she moves, when the player
asks for human thinking time.

It imports the unmodified `maia3` package and changes only what `go` prints. The
move is sampled exactly as `maia3-uci` samples it; the extra line is

```
info string policy top <p> chosen <q> legal <n>
```

with `<p>` the model's probability for its most likely move, `<q>` its probability
for the move it played (both in permille) and `<n>` the number of legal moves.
Unlike `maia3-uci`, candidates are not scored with the value head, so `info` lines
carry no score and a move takes about half the time.

## Install and run

```
pip install git+https://github.com/CSSLab/maia3.git
pip install .
maia3-policy-uci --model maia3-79m --use-uci-history
```

Arguments and UCI options (`Elo`, `SelfElo`, `OppoElo`, `Temperature`, `TopP`,
`MultiPV`) are those of `maia3-uci`. Tested with maia3 at commit
`1e13597c42d4858b7cfd7cfdae01e297263364b2` and the `UofTCSSLab/Maia3-79M` weights.

## License

GNU Affero General Public License v3.0 or later, the same as Maia 3. See
[LICENSE](LICENSE).
