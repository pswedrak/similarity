# similarity

## Measuring Semantic Similarity in the Taxonomy of Wordnet (sim max b)
### Finding graph
Currently, taxonomy is browsed breathly to find shorthest path between concepts by finding common concept, which is one of _hypernym_, _holonym_ or _meronym_.
All of them are treated as the same ones in criterion of similarity.

### Output

For `sim_max_b('cat', 'dog')` output is:
`PATHLEN:  4`, what is sum of `2+2`. Common concepts are both `Synset('carnivore.n.01')` and `Synset('male.n.02')`,
that are on distance `2` from `dog` synsets as well as distance `2` from `cat` synsets 

`At the distance of 1 from 'cat': [Synset('feline.n.01'), Synset('man.n.01'), ... , Synset('flog.v.01'), Synset('excrete.v.01')]`

`At the distance of 2 from 'cat': [Synset('carnivore.n.01'), ... , Synset('male.n.02'), Synset('communicator.n.01'), ... , Synset('imaging.n.02'), Synset('beat.v.02'), Synset('exhaust.v.05')]`

`At the distance of 1 from 'dog': [Synset('canine.n.02'), Synset('domestic_animal.n.01'), ... , Synset('hotdog.n.02'), Synset('ratchet.n.01')]`

`At the distance of 2 from 'dog': [Synset('carnivore.n.01'), ... , Synset('male.n.02'), Synset('unwelcome_person.n.01'), ... , Synset('mechanical_device.n.01'), Synset('spiral_ratchet_screwdriver.n.01')]`
