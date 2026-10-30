"""Maia 3 as a UCI engine that also says how sure the model was of its move.

This program imports the unmodified `maia3` package (https://github.com/CSSLab/maia3)
and changes only what `go` prints. It takes the same command-line arguments and
UCI options as `maia3-uci`.

On `go`, it samples a move exactly as `maia3-uci` does (move head, legal moves
only, the Temperature and TopP options), then prints:

    info depth 1 multipv <k> pv <move>          for the model's top MultiPV moves
    info string policy top <p> chosen <q> legal <n>
    bestmove <move>

where <p> is the model's probability for its most likely move and <q> its
probability for the move it played, both in permille, and <n> is the number of
legal moves. Unlike `maia3-uci` it does not score the candidates with the value
head, so `info` lines carry no score; the move itself is sampled the same way.

Copyright (C) 2026 Resablic LLC

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE file.
"""

import torch
from torch.amp import autocast

from maia3.dataset import get_legal_moves_mask
from maia3.uci import Maia3UCIEngine, parse_args, sample_from_logits
from maia3.utils import seed_everything


def permille(probability):
    return max(0, min(1000, round(float(probability) * 1000)))


class PolicyUCIEngine(Maia3UCIEngine):

    @torch.no_grad()
    def policy_move(self):
        """The sampled move, the top MultiPV moves, and the policy facts; None when there is no move."""
        if self.board.is_game_over():
            return None
        legal_mask = get_legal_moves_mask(self.board, self.all_moves_dict)
        legal = int(legal_mask.sum().item())
        if legal == 0:
            return None

        tokens = self._tokens_from_history(self.history).unsqueeze(0).to(self.cfg.device)
        self_elos = torch.tensor([self.self_elo], dtype=torch.long, device=self.cfg.device)
        oppo_elos = torch.tensor([self.oppo_elo], dtype=torch.long, device=self.cfg.device)
        with autocast('cuda', enabled=self.cfg.use_amp and self.cfg.device.startswith('cuda')):
            logits_move, _logits_value, _ = self.model(tokens, self_elos, oppo_elos)

        logits = logits_move[0].float().masked_fill(~legal_mask.to(self.cfg.device), float('-inf'))
        idx = sample_from_logits(logits, self.temperature, self.top_p)
        move = self._move_from_index(idx)
        if move is None:
            return None

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idxs = torch.topk(probs, k=min(self.multipv, legal))
        top_moves = [m for m in (self._move_from_index(i) for i in top_idxs.tolist()) if m is not None]
        return {
            "move": move,
            "top_moves": top_moves,
            "top": permille(top_probs[0]),
            "chosen": permille(probs[idx]),
            "legal": legal,
        }

    def cmd_go(self, line):
        self.ensure_model_loaded()
        result = self.policy_move()
        if result is None:
            move = None
        else:
            move = result["move"]
            for rank, top_move in enumerate(result["top_moves"], start=1):
                print(f"info depth 1 multipv {rank} pv {top_move.uci()}", flush=True)
            print(f"info string policy top {result['top']} chosen {result['chosen']} legal {result['legal']}", flush=True)

        if "infinite" in line.split():
            self.pending_bestmove = move
            self.pending_search = True
            return
        self.print_bestmove(move)


def main(argv=None):
    cfg = parse_args(argv)
    seed_everything(cfg.seed)
    PolicyUCIEngine(cfg).run()


if __name__ == "__main__":
    main()
