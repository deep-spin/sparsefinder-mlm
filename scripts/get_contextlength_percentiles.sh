cat longformer_stuff/triviaqa_longformer_data/triviaqa/squad_format/squad-wikipedia-dev-4096.json | grep '^                    "context":' | awk '{print NF - 1}' | sort -n | awk '{all[NR] = $0} END{print all[1], all[int(NR*0.25)], all[int(NR*0.5)], all[int(NR*0.75)], all[NR]}'

# will greatly overestimate, since it includes question, answer, etc:
cat longformer_stuff/wikihop/qangaroo_v1.1/wikihop/dev.tokenized.json | awk 'BEGIN{FS="\"supports\": "}{for (i=1 ; i <= NF; i++) print split($i,a," ")}' | sort -n | awk '{all[NR] = $0} END{print all[1], all[int(NR*0.25)], all[int(NR*0.5)], all[int(NR*0.75)], all[NR]}'
#improved:
echo 'oiejr rf rf "supports": foje rw "], er er "supports": rw qrt q tj "], ert  "supports": rew "], we ' | awk 'BEGIN{FS="\"supports\": "}{for (i=2 ; i <= NF; i++) {split($i,a," ") ; key=1; tokens=0; while (key<=length(a) && a[key]!="\"],") {++key; ++tokens} ; print tokens }}'
