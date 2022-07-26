#!/usr/bin/env python3

from itertools import chain

from spacy.matcher import DependencyMatcher

from .base import BaseAnnotator


class CoreferenceAnnotator(BaseAnnotator):
    def __init__(self, nlp):
        super().__init__(nlp)
        self.matcher = DependencyMatcher(nlp.vocab)

        # some/all/any/types/variety/categories of information
        pattern = [
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": "of", "POS": "ADP"}
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "coref_token",
                "RIGHT_ATTRS": {
                    "LEMMA": {"IN": ["some", "all", "any", "type", "variety", "category", "example"]},
                    "POS": {"IN": ["NOUN", "PRON"]}
                }
            },
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "main_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("COREF_SOME_OF", [pattern])

    def annotate_one_doc(self, document, doc):
        def infer_type(token):
            """Infer noun phrase type through SUBSUM edges"""
            # Use BFS here to avoid a loop.
            bfs_queue = [token]
            seen = {token}
            i = 0

            while i < len(bfs_queue):
                tok = bfs_queue[i]
                i += 1

                if tok._.ent_type in ["DATA", "ACTOR"]:
                    return tok._.ent_type
                elif tok._.ent_type == "NN":
                    for _, linked_token, relationship in document.get_all_links(tok):
                        if relationship in ["SUBSUM", "COREF"] and linked_token not in seen:
                            bfs_queue.append(linked_token)
                            seen.add(linked_token)

            return None

        last_sentence_ents = []

        # Handle pronouns
        for sent in doc.sents:
            current_sentence_ents = []

            for noun_phrase in sent.ents:
                referent = None

                if (noun_phrase[0].lemma_ in {"this", "that", "these", "those"}
                    and noun_phrase[0].head == noun_phrase[-1]):
                    # Resolve this/that/these/those xxx
                    for prev_noun_phrase in chain(reversed(current_sentence_ents), reversed(last_sentence_ents)):
                        if prev_noun_phrase[-1].lemma_ == noun_phrase[-1].lemma_:
                            referent = prev_noun_phrase
                            reason = "SAME_ROOT"
                            break

                if referent is None and noun_phrase.lemma_ in ["they", "this", "these", "it"]:
                    inferred_type = infer_type(noun_phrase.root)

                    for prev_noun_phrase in chain(reversed(current_sentence_ents), reversed(last_sentence_ents)):
                        if prev_noun_phrase._.ent_type == inferred_type:
                            referent = prev_noun_phrase
                            reason = "SAME_TYPE"
                            break

                if referent is None and noun_phrase.lemma_ == "they":
                    # Resolve "they" (referring to an entity)
                    for prev_noun_phrase in chain(reversed(current_sentence_ents), reversed(last_sentence_ents)):
                        if (prev_noun_phrase._.ent_type == "ACTOR" and prev_noun_phrase.root.tag_ in ["NNS", "NNPS"]):
                            referent = prev_noun_phrase
                            reason = "THEY_ACTOR"
                            break

                if referent is not None:
                    self.logger.info("Sentence 1: %r", referent.sent.text)
                    self.logger.info("Sentence 2: %r", noun_phrase.sent.text)
                    self.logger.info("Edge COREF (%s): %r -> %r", reason, noun_phrase.text, referent.text)

                    document.link(noun_phrase.root, referent.root, "COREF")

                current_sentence_ents.append(noun_phrase)

            last_sentence_ents = current_sentence_ents

        # Handle special patterns
        for match_id, matched_tokens in self.matcher(doc):
            rule_name = self.vocab.strings[match_id]
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            coref_noun_phrase = match_info["coref_token"]._.ent
            main_noun_phrase = match_info["main_token"]._.ent

            if coref_noun_phrase is None or main_noun_phrase is None or coref_noun_phrase == main_noun_phrase:
                continue

            self.logger.info("Rule %s matches %r", rule_name, main_noun_phrase.sent.text)
            self.logger.info("Edge COREF: %r -> %r", coref_noun_phrase.text, main_noun_phrase.text)
            document.link(coref_noun_phrase.root, main_noun_phrase.root, "COREF")

    def annotate(self, document):
        for doc in document.iter_docs():
            self.annotate_one_doc(document, doc)
