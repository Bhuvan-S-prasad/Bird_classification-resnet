const birdData = {
    "001.Black_footed_Albatross": {
      id: "001",
      name: "Black-footed Albatross",
      scientificName: "Phoebastria nigripes",
      image: "https://i.postimg.cc/635b8VkX/Black-Footed-Albatross-0002-55.jpg",
      habitat: "Open ocean of the North Pacific; nests on remote islands",
      diet: "Squid, fish, and fish eggs. Often follows fishing vessels for discarded fish",
      lifespan: "12-40 years",
      description: "A large seabird with dark plumage and distinctive black feet. Known for their incredible ability to glide over ocean waves for hours without flapping their wings."
    },
    "002.Laysan_Albatross": {
      id: "002",
      name: "Laysan Albatross",
      scientificName: "Phoebastria immutabilis",
      image: "https://images.unsplash.com/photo-1594733836693-0b51372ae7f4?auto=format&fit=crop&q=80",
      habitat: "Open ocean of the North Pacific; breeds primarily on Hawaiian islands",
      diet: "Squid, fish, and crustaceans",
      lifespan: "Up to 40 years",
      description: "Distinguished by its white body and dark wings, known for their elaborate courtship dances and strong pair bonds that can last for life."
    },
    "003.Sooty_Albatross": {
    id: "003",
    name: "Sooty Albatross",
    scientificName: "Phoebetria fusca",
    image: "https://images.unsplash.com/photo-1552727451-6f5671e14d83?auto=format&fit=crop&q=80",
    habitat: "Southern oceans, particularly around sub-Antarctic islands",
    diet: "Squid, fish, and marine crustaceans",
    lifespan: "Up to 35 years",
    description: "Dark-plumaged seabird with distinctive white eye-rings and yellow stripe on the bill. Expert glider in strong winds."
  },
  "004.Groove_billed_Ani": {
    id: "004",
    name: "Groove-billed Ani",
    scientificName: "Crotophaga sulcirostris",
    image: "https://images.unsplash.com/photo-1589595363745-d842812a9db7?auto=format&fit=crop&q=80",
    habitat: "Open and semi-open areas, farmland, and scrubby woodland",
    diet: "Insects, lizards, and fruits",
    lifespan: "4-8 years",
    description: "Social bird with distinctive grooved bill, often seen in small groups. Known for cooperative breeding where several females lay eggs in one nest."
  },
  "005.Crested_Auklet": {
    id: "005",
    name: "Crested Auklet",
    scientificName: "Aethia cristatella",
    image: "https://images.unsplash.com/photo-1550853024-fae8cd4be47f?auto=format&fit=crop&q=80",
    habitat: "North Pacific Ocean and Bering Sea",
    diet: "Plankton and small marine invertebrates",
    lifespan: "Up to 20 years",
    description: "Distinguished by its forward-curving crest and orange bill. Known for their citrus-like scent during breeding season."
  },
  "006.Least_Auklet": {
    id: "006",
    name: "Least Auklet",
    scientificName: "Aethia pusilla",
    image: "https://images.unsplash.com/photo-1552083375-1447ce886485?auto=format&fit=crop&q=80",
    habitat: "Bering Sea and North Pacific Ocean",
    diet: "Small marine crustaceans and plankton",
    lifespan: "Up to 12 years",
    description: "Smallest diving bird in North America. Recognizable by their small size and white facial plumes during breeding season."
  },
  "007.Parakeet_Auklet": {
    id: "007",
    name: "Parakeet Auklet",
    scientificName: "Aethia psittacula",
    image: "https://images.unsplash.com/photo-1591608971362-f08b2a75731a?auto=format&fit=crop&q=80",
    habitat: "North Pacific Ocean and Bering Sea",
    diet: "Small fish, squid, and marine invertebrates",
    lifespan: "15-20 years",
    description: "Medium-sized seabird with distinctive upturned bill and white facial plumes. Excellent underwater swimmer."
  },
  "008.Rhinoceros_Auklet": {
    id: "008",
    name: "Rhinoceros Auklet",
    scientificName: "Cerorhinca monocerata",
    image: "https://images.unsplash.com/photo-1550853123-fe1a3e5e1b5d?auto=format&fit=crop&q=80",
    habitat: "Coastal waters of the North Pacific",
    diet: "Small fish and squid",
    lifespan: "Up to 30 years",
    description: "Named for the horn-like projection on its bill during breeding season. Nocturnal on breeding colonies."
  },
  "009.Brewer_Blackbird": {
    id: "009",
    name: "Brewer's Blackbird",
    scientificName: "Euphagus cyanocephalus",
    image: "https://images.unsplash.com/photo-1589595363746-d842812a9db8?auto=format&fit=crop&q=80",
    habitat: "Open and semi-open areas, urban parks, and farmland",
    diet: "Insects, seeds, and grains",
    lifespan: "12-15 years",
    description: "Glossy black bird with piercing yellow eyes. Highly adaptable to urban environments."
  },
  "010.Red_winged_Blackbird": {
    id: "010",
    name: "Red-winged Blackbird",
    scientificName: "Agelaius phoeniceus",
    image: "https://images.unsplash.com/photo-1589595363747-d842812a9db9?auto=format&fit=crop&q=80",
    habitat: "Marshes, wetlands, and grasslands",
    diet: "Insects, seeds, and grains",
    lifespan: "15-20 years",
    description: "Males are black with distinctive red and yellow shoulder patches. One of the most abundant birds in North America."
  },
  "011.Rusty_Blackbird": {
    id: "011",
    name: "Rusty Blackbird",
    scientificName: "Euphagus carolinus",
    image: "https://images.unsplash.com/photo-1604083992781-82b665f6c33e?auto=format&fit=crop&q=80",
    habitat: "Boreal forests, wetlands, and marshes",
    diet: "Insects, seeds, and small aquatic invertebrates",
    lifespan: "8-12 years",
    description: "Medium-sized blackbird with rusty-brown plumage in winter. Known for their sharp, creaky song."
  },
  "012.Yellow_headed_Blackbird": {
    id: "012",
    name: "Yellow-headed Blackbird",
    scientificName: "Xanthocephalus xanthocephalus",
    image: "https://images.unsplash.com/photo-1614530141169-83c84e1bc3a3?auto=format&fit=crop&q=80",
    habitat: "Freshwater marshes and wetlands",
    diet: "Insects and seeds",
    lifespan: "10-15 years",
    description: "Striking bird with bright yellow head and chest. Found in large flocks during migration."
  },
  "013.Bobolink": {
    id: "013",
    name: "Bobolink",
    scientificName: "Dolichonyx oryzivorus",
    image: "https://images.unsplash.com/photo-1612534894898-e5b7c2d7b9fa?auto=format&fit=crop&q=80",
    habitat: "Grasslands and prairies",
    diet: "Seeds, grains, and insects",
    lifespan: "4-9 years",
    description: "Known for their bubbly, metallic song. Males have distinctive black and white plumage during the breeding season."
  },
  "014.Indigo_Bunting": {
    id: "014",
    name: "Indigo Bunting",
    scientificName: "Passerina cyanea",
    image: "https://images.unsplash.com/photo-1552427303-1a61f9d6471d?auto=format&fit=crop&q=80",
    habitat: "Open woodlands, farmlands, and forest edges",
    diet: "Insects, seeds, and berries",
    lifespan: "5-10 years",
    description: "Brilliant blue male plumage during summer. Females are brownish and well-camouflaged."
  },
  "015.Lazuli_Bunting": {
    id: "015",
    name: "Lazuli Bunting",
    scientificName: "Passerina amoena",
    image: "https://images.unsplash.com/photo-1597765905451-917dc2de1f78?auto=format&fit=crop&q=80",
    habitat: "Shrubby areas, open woodlands, and riparian zones",
    diet: "Seeds, insects, and fruits",
    lifespan: "3-8 years",
    description: "Male has vivid blue plumage with an orange breast. Known for their melodious song."
  },
  "016.Painted_Bunting": {
    id: "016",
    name: "Painted Bunting",
    scientificName: "Passerina ciris",
    image: "https://images.unsplash.com/photo-1602854364251-97e97d678ea7?auto=format&fit=crop&q=80",
    habitat: "Woodland edges, thickets, and brushy areas",
    diet: "Seeds and insects",
    lifespan: "5-8 years",
    description: "One of the most colorful songbirds in North America. Males are a vivid blend of blue, green, and red."
  },
  "017.Cardinal": {
    id: "017",
    name: "Northern Cardinal",
    scientificName: "Cardinalis cardinalis",
    image: "https://images.unsplash.com/photo-1589750080671-cda3eecac48c?auto=format&fit=crop&q=80",
    habitat: "Woodlands, gardens, and shrublands",
    diet: "Seeds, fruits, and insects",
    lifespan: "3-15 years",
    description: "Bright red males are a familiar sight in backyards. Females are tan with reddish tinges."
  },
  "018.Spotted_Catbird": {
    id: "018",
    name: "Spotted Catbird",
    scientificName: "Ailuroedus maculosus",
    image: "https://images.unsplash.com/photo-1596547895033-1ae841f54c0d?auto=format&fit=crop&q=80",
    habitat: "Rainforests and dense woodlands",
    diet: "Fruits and insects",
    lifespan: "Up to 10 years",
    description: "Named for their cat-like calls. Green plumage with spots gives them a unique appearance."
  },
  "019.Gray_Catbird": {
    id: "019",
    name: "Gray Catbird",
    scientificName: "Dumetella carolinensis",
    image: "https://images.unsplash.com/photo-1613398755774-f632b36fdf6f?auto=format&fit=crop&q=80",
    habitat: "Thickets, forests, and suburban areas",
    diet: "Insects, fruits, and berries",
    lifespan: "4-10 years",
    description: "Sleek gray bird with a distinctive black cap. Known for their varied song and mimicry."
  },
  "020.Yellow_breasted_Chat": {
    id: "020",
    name: "Yellow-breasted Chat",
    scientificName: "Icteria virens",
    image: "https://images.unsplash.com/photo-1615074852898-2dd28574f9d8?auto=format&fit=crop&q=80",
    habitat: "Thickets, shrubs, and forest edges",
    diet: "Insects, fruits, and berries",
    lifespan: "Up to 10 years",
    description: "Largest warbler in North America, with a bright yellow breast and bold behavior."
  },
  "021.Eastern_Towhee": {
    id: "021",
    name: "Eastern Towhee",
    scientificName: "Pipilo erythrophthalmus",
    image: "https://images.unsplash.com/photo-1615176672595-235e3bca6ff4?auto=format&fit=crop&q=80",
    habitat: "Shrubby areas, woodlands, and forest edges",
    diet: "Seeds, fruits, and insects",
    lifespan: "4-9 years",
    description: "Distinctive black, white, and rufous plumage. Known for their sharp 'drink your tea' call."
  },
  "022.Chuck_will_Widow": {
      id: "022",
      name: "Chuck-will's-widow",
      scientificName: "Antrostomus carolinensis",
      image: "https://images.unsplash.com/photo-1602732558362-e88fd1a1535d?auto=format&fit=crop&q=80",
      habitat: "Open woodlands and forest edges",
      diet: "Insects, especially moths and beetles",
      lifespan: "Up to 5 years",
      description: "A nocturnal bird with long wings and a distinctive churring call. Related to the whip-poor-will."
    },
    "023.Brandt_Cormorant": {
      id: "023",
      name: "Brandt's Cormorant",
      scientificName: "Phalacrocorax penicillatus",
      image: "https://images.unsplash.com/photo-1573485136934-72e7a3b59453?auto=format&fit=crop&q=80",
      habitat: "Coastal cliffs and rocky shores",
      diet: "Fish, crustaceans, and invertebrates",
      lifespan: "Up to 13 years",
      description: "A large, dark seabird with a striking blue throat and long, pointed bill."
    },
    "024.Red_faced_Cormorant": {
      id: "024",
      name: "Red-faced Cormorant",
      scientificName: "Phalacrocorax urile",
      image: "https://images.unsplash.com/photo-1564849468-b9a1192e1be0?auto=format&fit=crop&q=80",
      habitat: "Rocky coastlines and islands",
      diet: "Fish, squid, and other marine life",
      lifespan: "Up to 8 years",
      description: "A striking seabird with a bright red face and sharp bill, known for diving to catch prey."
    },
    "025.Pelagic_Cormorant": {
      id: "025",
      name: "Pelagic Cormorant",
      scientificName: "Phalacrocorax pelagicus",
      image: "https://images.unsplash.com/photo-1585538762479-e2a88186b2ca?auto=format&fit=crop&q=80",
      habitat: "Coastal waters and rocky islands",
      diet: "Fish and invertebrates",
      lifespan: "Up to 12 years",
      description: "A small, dark seabird with a slender body and striking greenish-black plumage."
    },
    "026.Bronzed_Cowbird": {
      id: "026",
      name: "Bronzed Cowbird",
      scientificName: "Molothrus aeneus",
      image: "https://images.unsplash.com/photo-1619198333091-7893b61561d9?auto=format&fit=crop&q=80",
      habitat: "Open grasslands and agricultural areas",
      diet: "Seeds, grains, and insects",
      lifespan: "Up to 8 years",
      description: "A shiny, dark-colored bird often seen in large flocks, known for parasitizing other birds' nests."
    },
    "027.Shiny_Cowbird": {
      id: "027",
      name: "Shiny Cowbird",
      scientificName: "Molothrus bonariensis",
      image: "https://images.unsplash.com/photo-1597835400384-9a15f1c4a230?auto=format&fit=crop&q=80",
      habitat: "Open fields, pastures, and agricultural areas",
      diet: "Insects, seeds, and grains",
      lifespan: "Up to 8 years",
      description: "A glossy, black bird with a short bill, often seen near livestock or human settlements."
    },
    "028.Brown_Creeper": {
      id: "028",
      name: "Brown Creeper",
      scientificName: "Certhia americana",
      image: "https://images.unsplash.com/photo-1607079227425-1da5876630bc?auto=format&fit=crop&q=80",
      habitat: "Forest habitats, especially coniferous trees",
      diet: "Insects and spiders found on tree bark",
      lifespan: "Up to 7 years",
      description: "A small, cryptic bird with an upward-curved bill, often seen creeping along tree trunks."
    },
    "029.American_Crow": {
      id: "029",
      name: "American Crow",
      scientificName: "Corvus brachyrhynchos",
      image: "https://images.unsplash.com/photo-1590860520246-0c75f44f66d0?auto=format&fit=crop&q=80",
      habitat: "A wide range of habitats, from forests to urban areas",
      diet: "Omnivorous, including insects, fruits, and small animals",
      lifespan: "Up to 8 years",
      description: "A highly intelligent bird known for its black feathers and loud, harsh call. Often seen in large groups."
    },
    "030.Fish_Crow": {
      id: "030",
      name: "Fish Crow",
      scientificName: "Corvus ossifragus",
      image: "https://images.unsplash.com/photo-1573146846655-93a2bbde3d7e?auto=format&fit=crop&q=80",
      habitat: "Coastal areas, wetlands, and estuaries",
      diet: "Fish, crabs, and other small aquatic animals",
      lifespan: "Up to 7 years",
      description: "A smaller, more specialized cousin of the American Crow, known for its preference for coastal habitats."
    },
  "031.Black_billed_Cuckoo": {
    id: "031",
    name: "Black-billed Cuckoo",
    scientificName: "Coccyzus erythropthalmus",
    image: "https://images.unsplash.com/photo-1594475860892-cf033f2172d2?auto=format&fit=crop&q=80",
    habitat: "Deciduous forests, woodlands, and suburban areas",
    diet: "Insects, especially caterpillars",
    lifespan: "Up to 10 years",
    description: "A medium-sized bird known for its black bill and distinctive, cuckoo-like call."
  },
  "032.Mangrove_Cuckoo": {
    id: "032",
    name: "Mangrove Cuckoo",
    scientificName: "Coccyzus minor",
    image: "https://images.unsplash.com/photo-1555953082-b63e17a26cd7?auto=format&fit=crop&q=80",
    habitat: "Mangrove forests and coastal areas",
    diet: "Insects and small vertebrates",
    lifespan: "Up to 8 years",
    description: "A secretive, large cuckoo with a long tail and distinct call, often found in coastal mangroves."
  },
  "033.Yellow_billed_Cuckoo": {
    id: "033",
    name: "Yellow-billed Cuckoo",
    scientificName: "Coccyzus americanus",
    image: "https://images.unsplash.com/photo-1556166430-0c618c2184ad?auto=format&fit=crop&q=80",
    habitat: "Woodlands, forests, and riparian areas",
    diet: "Insects, particularly caterpillars",
    lifespan: "Up to 12 years",
    description: "A bird with a bright yellow bill, known for its distinct call and insectivorous diet."
  },
  "034.Gray_crowned_Rosy_Finch": {
    id: "034",
    name: "Gray-crowned Rosy Finch",
    scientificName: "Leucosticte tephrocotis",
    image: "https://images.unsplash.com/photo-1582251600840-32b5fa1c7eb4?auto=format&fit=crop&q=80",
    habitat: "Alpine and mountainous regions",
    diet: "Seeds, insects, and berries",
    lifespan: "Up to 6 years",
    description: "A small, high-altitude finch with a gray crown and rosy pink plumage."
  },
  "035.Purple_Finch": {
    id: "035",
    name: "Purple Finch",
    scientificName: "Haemorhous purpureus",
    image: "https://images.unsplash.com/photo-1581454543209-e51a49f98e4f?auto=format&fit=crop&q=80",
    habitat: "Woodlands, coniferous forests, and suburbs",
    diet: "Seeds, fruits, and insects",
    lifespan: "Up to 7 years",
    description: "A medium-sized finch with deep red plumage and a distinctive song."
  },
  "036.Northern_Flicker": {
    id: "036",
    name: "Northern Flicker",
    scientificName: "Colaptes auratus",
    image: "https://images.unsplash.com/photo-1588086842214-299a5132328d?auto=format&fit=crop&q=80",
    habitat: "Wooded areas, including forests and urban parks",
    diet: "Ants, beetles, and other insects",
    lifespan: "Up to 9 years",
    description: "A large woodpecker known for its distinctive call and brownish plumage with black spots."
  },
  "037.Acadian_Flycatcher": {
    id: "037",
    name: "Acadian Flycatcher",
    scientificName: "Empidonax virescens",
    image: "https://images.unsplash.com/photo-1595186092251-30b9189d377f?auto=format&fit=crop&q=80",
    habitat: "Wooded areas, especially near streams",
    diet: "Insects, especially flying insects",
    lifespan: "Up to 5 years",
    description: "A small flycatcher with olive-green plumage and a distinctive 'fitz-bew' call."
  },
  "038.Great_Crested_Flycatcher": {
    id: "038",
    name: "Great Crested Flycatcher",
    scientificName: "Myiarchus crinitus",
    image: "https://images.unsplash.com/photo-1565471532-5ea57d539c5e?auto=format&fit=crop&q=80",
    habitat: "Woodlands, forests, and suburban areas",
    diet: "Insects, berries, and small fruits",
    lifespan: "Up to 10 years",
    description: "A medium-sized flycatcher with a large crest and a yellow belly."
  },
  "039.Least_Flycatcher": {
    id: "039",
    name: "Least Flycatcher",
    scientificName: "Empidonax minimus",
    image: "https://images.unsplash.com/photo-1589412651232-1e20d575b404?auto=format&fit=crop&q=80",
    habitat: "Wooded areas, particularly near streams",
    diet: "Insects, especially small flying insects",
    lifespan: "Up to 7 years",
    description: "A small, olive-green flycatcher known for its sharp 'che-bek' call."
  },
  "040.Olive_sided_Flycatcher": {
    id: "040",
    name: "Olive-sided Flycatcher",
    scientificName: "Contopus cooperi",
    image: "https://images.unsplash.com/photo-1594356429568-3c88d01f999f?auto=format&fit=crop&q=80",
    habitat: "Forests, especially in the western U.S.",
    diet: "Insects, particularly large flying insects",
    lifespan: "Up to 8 years",
    description: "A medium-sized flycatcher with olive-green plumage and a distinctive 'quick, three beers' call."
  },
  "041.Scissor_tailed_Flycatcher": {
    id: "041",
    name: "Scissor-tailed Flycatcher",
    scientificName: "Tyrannus forficatus",
    image: "https://images.unsplash.com/photo-1566441297-c18a253db2b5?auto=format&fit=crop&q=80",
    habitat: "Open woodlands and prairies",
    diet: "Insects, especially flying insects",
    lifespan: "Up to 6 years",
    description: "A striking flycatcher with long, scissor-like tail feathers and a pale underbelly."
  },
  "042.Vermilion_Flycatcher": {
    id: "042",
    name: "Vermilion Flycatcher",
    scientificName: "Pyrocephalus rubinus",
    image: "https://images.unsplash.com/photo-1588587343761-c3b09e516f93?auto=format&fit=crop&q=80",
    habitat: "Open woodlands, riparian areas, and deserts",
    diet: "Insects, especially beetles and ants",
    lifespan: "Up to 5 years",
    description: "A small, striking flycatcher with brilliant vermilion plumage and dark wings."
  },
  "043.Yellow_bellied_Flycatcher": {
    id: "043",
    name: "Yellow-bellied Flycatcher",
    scientificName: "Empidonax flaviventris",
    image: "https://images.unsplash.com/photo-1585083021322-6352c6f72256?auto=format&fit=crop&q=80",
    habitat: "Wooded areas, especially near water",
    diet: "Insects, especially moths and beetles",
    lifespan: "Up to 5 years",
    description: "A small flycatcher with a yellow belly and olive-green plumage, known for its soft 'che-bek' call."
  },
  "044.Frigatebird": {
    id: "044",
    name: "Frigatebird",
    scientificName: "Fregata magnificens",
    image: "https://images.unsplash.com/photo-1585330490971-4734a68a5408?auto=format&fit=crop&q=80",
    habitat: "Tropical coastal regions and islands",
    diet: "Fish, squid, and other marine life",
    lifespan: "Up to 15 years",
    description: "A large seabird with a forked tail and a distinctive, red throat pouch."
  },
  "045.Northern_Fulmar": {
    id: "045",
    name: "Northern Fulmar",
    scientificName: "Fulmarus glacialis",
    image: "https://images.unsplash.com/photo-1556076995-75f33a6842e1?auto=format&fit=crop&q=80",
    habitat: "Coastal cliffs and tundra",
    diet: "Fish, squid, and krill",
    lifespan: "Up to 40 years",
    description: "A seabird with a heavy bill, known for its ability to glide over the ocean for long distances."
  },
  "046.Gadwall": {
    id: "046",
    name: "Gadwall",
    scientificName: "Mareca strepera",
    image: "https://images.unsplash.com/photo-1561012422-9e062681ffaf?auto=format&fit=crop&q=80",
    habitat: "Wetlands, lakes, and rivers",
    diet: "Aquatic plants, seeds, and small invertebrates",
    lifespan: "Up to 10 years",
    description: "A medium-sized dabbling duck with a distinctive grayish-brown plumage and a white belly."
  },
  "047.American_Goldfinch": {
    id: "047",
    name: "American Goldfinch",
    scientificName: "Spinus tristis",
    image: "https://images.unsplash.com/photo-1570711865954-d4fffa9a5db1?auto=format&fit=crop&q=80",
    habitat: "Fields, gardens, and suburban areas",
    diet: "Seeds, especially sunflower seeds",
    lifespan: "Up to 11 years",
    description: "A small finch with bright yellow plumage in males, known for its cheerful, twitting call."
  },
  "048.European_Goldfinch": {
    id: "048",
    name: "European Goldfinch",
    scientificName: "Carduelis carduelis",
    image: "https://images.unsplash.com/photo-1577277986581-8b5c8c37b3c5?auto=format&fit=crop&q=80",
    habitat: "Woodlands, gardens, and farmland",
    diet: "Seeds, particularly thistle and dandelion seeds",
    lifespan: "Up to 9 years",
    description: "A colorful finch with a red face, yellow wing bars, and a distinctive trilling song."
  },
  "049.Boat_tailed_Grackle": {
    id: "049",
    name: "Boat-tailed Grackle",
    scientificName: "Quiscalus major",
    image: "https://images.unsplash.com/photo-1568654111-3b2c5e0f6e61?auto=format&fit=crop&q=80",
    habitat: "Coastal marshes, salt flats, and wetlands",
    diet: "Insects, seeds, and small vertebrates",
    lifespan: "Up to 9 years",
    description: "A large blackbird with an iridescent body and a long, boat-shaped tail."
  },
  "050.Eared_Grebe": {
    id: "050",
    name: "Eared Grebe",
    scientificName: "Podiceps nigricollis",
    image: "https://images.unsplash.com/photo-1561955224-4f550fa1d77b?auto=format&fit=crop&q=80",
    habitat: "Lakes and marshes",
    diet: "Small fish, aquatic insects, and crustaceans",
    lifespan: "Up to 5 years",
    description: "A small waterbird with dark plumage and distinctive, ear-like tufts during the breeding season."
  },
  "051.Horned_Grebe": {
    id: "051",
    name: "Horned Grebe",
    scientificName: "Podiceps auritus",
    image: "https://images.unsplash.com/photo-1579147400836-401cc29fa90b?auto=format&fit=crop&q=80",
    habitat: "Lakes, ponds, and marshes",
    diet: "Small fish, aquatic insects, and crustaceans",
    lifespan: "Up to 5 years",
    description: "A small waterbird with a distinctive black and white plumage and tufted horns during the breeding season."
  },
  "052.Pied_billed_Grebe": {
    id: "052",
    name: "Pied-billed Grebe",
    scientificName: "Podilymbus podiceps",
    image: "https://images.unsplash.com/photo-1572857543279-9a5c2e2b5f40?auto=format&fit=crop&q=80",
    habitat: "Lakes, ponds, and marshes",
    diet: "Small fish, amphibians, and aquatic invertebrates",
    lifespan: "Up to 10 years",
    description: "A small grebe with a distinctive black ring around its bill and a chunky, rounded body."
  },
  "053.Western_Grebe": {
    id: "053",
    name: "Western Grebe",
    scientificName: "Aechmophorus occidentalis",
    image: "https://images.unsplash.com/photo-1586281552076-540a45b72d0b?auto=format&fit=crop&q=80",
    habitat: "Large lakes and reservoirs",
    diet: "Fish and aquatic invertebrates",
    lifespan: "Up to 7 years",
    description: "A large grebe with striking black and white plumage and a slender, long neck."
  },
  "054.Blue_Grosbeak": {
    id: "054",
    name: "Blue Grosbeak",
    scientificName: "Passerina cyanea",
    image: "https://images.unsplash.com/photo-1575702066192-01ae63c3cc77?auto=format&fit=crop&q=80",
    habitat: "Open woodlands and brushy fields",
    diet: "Seeds, grains, and insects",
    lifespan: "Up to 7 years",
    description: "A striking blue bird with a large, conical bill and bold, dark markings."
  },
  "055.Evening_Grosbeak": {
    id: "055",
    name: "Evening Grosbeak",
    scientificName: "Coccothraustes vespertinus",
    image: "https://images.unsplash.com/photo-1579335533605-2e79f073f3b9?auto=format&fit=crop&q=80",
    habitat: "Woodlands and coniferous forests",
    diet: "Seeds, fruits, and insects",
    lifespan: "Up to 13 years",
    description: "A large grosbeak with a yellow and black plumage, known for its powerful beak."
  },
  "056.Pine_Grosbeak": {
    id: "056",
    name: "Pine Grosbeak",
    scientificName: "Pinicola enucleator",
    image: "https://images.unsplash.com/photo-1584562133176-205c62a6a03c?auto=format&fit=crop&q=80",
    habitat: "Coniferous forests and high altitudes",
    diet: "Fruits, seeds, and buds",
    lifespan: "Up to 10 years",
    description: "A robust grosbeak with a large bill and colorful plumage, often found in colder regions."
  },
  "057.Rose_breasted_Grosbeak": {
    id: "057",
    name: "Rose-breasted Grosbeak",
    scientificName: "Pheucticus ludovicianus",
    image: "https://images.unsplash.com/photo-1581572880982-218f4b9e43c5?auto=format&fit=crop&q=80",
    habitat: "Woodlands, forests, and gardens",
    diet: "Seeds, fruits, and insects",
    lifespan: "Up to 9 years",
    description: "A striking grosbeak with a distinctive rose-colored patch on its breast and a large, conical bill."
  },
  "058.Pigeon_Guillemot": {
    id: "058",
    name: "Pigeon Guillemot",
    scientificName: "Cepphus columba",
    image: "https://images.unsplash.com/photo-1575308254564-9d107635c70b?auto=format&fit=crop&q=80",
    habitat: "Coastal cliffs and rocky shores",
    diet: "Fish and invertebrates",
    lifespan: "Up to 15 years",
    description: "A seabird with black plumage and a distinctive white wing patch, often found in rocky coastal habitats."
  },
  "059.California_Gull": {
    id: "059",
    name: "California Gull",
    scientificName: "Larus californicus",
    image: "https://images.unsplash.com/photo-1573429522470-cdff89c984a4?auto=format&fit=crop&q=80",
    habitat: "Coastal areas, lakes, and inland wetlands",
    diet: "Fish, insects, and small invertebrates",
    lifespan: "Up to 30 years",
    description: "A medium-sized gull with a light gray back and yellow legs, known for its adaptability to various habitats."
  },
  "060.Glaucous_winged_Gull": {
    id: "060",
    name: "Glaucous-winged Gull",
    scientificName: "Larus glaucescens",
    image: "https://images.unsplash.com/photo-1564631545-1dfe321ce03a?auto=format&fit=crop&q=80",
    habitat: "Coastal areas, particularly along the Pacific coast",
    diet: "Fish, mollusks, and scavenged food",
    lifespan: "Up to 20 years",
    description: "A large gull with a pale gray back and white underparts, often seen in coastal environments."
  },
  "061.Heermann_Gull": {
    id: "061",
    name: "Heermann Gull",
    scientificName: "Larus heermanni",
    image: "https://images.unsplash.com/photo-1580464631188-f3a9e2b0a6da?auto=format&fit=crop&q=80",
    habitat: "Coastal beaches and cliffs",
    diet: "Fish, mollusks, and scavenged food",
    lifespan: "Up to 25 years",
    description: "A medium-sized gull with a distinctive dark red bill and pale gray plumage."
  },
  "062.Herring_Gull": {
    id: "062",
    name: "Herring Gull",
    scientificName: "Larus argentatus",
    image: "https://images.unsplash.com/photo-1604267425975-c937fc6e9c02?auto=format&fit=crop&q=80",
    habitat: "Coastal areas, lakes, and inland waters",
    diet: "Fish, invertebrates, and carrion",
    lifespan: "Up to 30 years",
    description: "A large gull with pale gray wings and a yellow bill, often seen in urban and coastal environments."
  },
  "063.Ivory_Gull": {
    id: "063",
    name: "Ivory Gull",
    scientificName: "Pagophila eburnea",
    image: "https://images.unsplash.com/photo-1573325484095-b4f246f7e5f1?auto=format&fit=crop&q=80",
    habitat: "Arctic regions, especially on sea ice",
    diet: "Fish, invertebrates, and carrion",
    lifespan: "Up to 15 years",
    description: "A pure white gull found in polar regions, often associated with sea ice and glaciers."
  },
  "064.Ring_billed_Gull": {
    id: "064",
    name: "Ring-billed Gull",
    scientificName: "Larus delawarensis",
    image: "https://images.unsplash.com/photo-1612283464633-16b241de39d9?auto=format&fit=crop&q=80",
    habitat: "Coastal areas, lakes, and inland waterways",
    diet: "Fish, insects, and garbage",
    lifespan: "Up to 20 years",
    description: "A medium-sized gull with a characteristic black ring around its bill and a versatile diet."
  },
  "065.Slaty_backed_Gull": {
    id: "065",
    name: "Slaty-backed Gull",
    scientificName: "Larus schistisagus",
    image: "https://images.unsplash.com/photo-1612967203175-566470edb15f?auto=format&fit=crop&q=80",
    habitat: "Coastal and inland areas, especially in northern Asia",
    diet: "Fish, invertebrates, and small mammals",
    lifespan: "Up to 25 years",
    description: "A large gull with a slaty gray back and a distinctive robust bill, primarily found in northern regions."
  },
  "066.Western_Gull": {
    id: "066",
    name: "Western Gull",
    scientificName: "Larus occidentalis",
    image: "https://images.unsplash.com/photo-1612312653804-84cbb7c2be58?auto=format&fit=crop&q=80",
    habitat: "Coastal beaches and cliffs along the Pacific coast",
    diet: "Fish, mollusks, and carrion",
    lifespan: "Up to 30 years",
    description: "A large gull with a dark gray back and a robust bill, commonly found along the Pacific coastline."
  },
  "067.Anna_Hummingbird": {
    id: "067",
    name: "Anna Hummingbird",
    scientificName: "Calypte anna",
    image: "https://images.unsplash.com/photo-1564471211-4edb2d93ad7a?auto=format&fit=crop&q=80",
    habitat: "Gardens, shrubs, and woodlands in the western U.S.",
    diet: "Nectar, small insects, and spiders",
    lifespan: "Up to 9 years",
    description: "A small hummingbird with vibrant magenta and green plumage, commonly found in western gardens."
  },
  "068.Ruby_throated_Hummingbird": {
    id: "068",
    name: "Ruby-throated Hummingbird",
    scientificName: "Archilochus colubris",
    image: "https://images.unsplash.com/photo-1560194084-f587c1e95f93?auto=format&fit=crop&q=80",
    habitat: "Woodlands, gardens, and forests in eastern North America",
    diet: "Nectar, small insects, and spiders",
    lifespan: "Up to 9 years",
    description: "A small hummingbird with an iridescent ruby-red throat and green back, one of the most common hummingbirds in North America."
  },
  "069.Rufous_Hummingbird": {
    id: "069",
    name: "Rufous Hummingbird",
    scientificName: "Selasphorus rufus",
    image: "https://images.unsplash.com/photo-1572483414723-d1c40874c67a?auto=format&fit=crop&q=80",
    habitat: "Gardens and forests in the western U.S. and Canada",
    diet: "Nectar, small insects, and spiders",
    lifespan: "Up to 5 years",
    description: "A small hummingbird with a fiery red throat and greenish back, known for its aggressive territorial behavior."
  },
  "070.Green_Violetear": {
    id: "070",
    name: "Green Violetear",
    scientificName: "Colibri thalassinus",
    image: "https://images.unsplash.com/photo-1564628171-274dbf8d3ec1?auto=format&fit=crop&q=80",
    habitat: "Montane forests and gardens in Central America",
    diet: "Nectar, small insects, and spiders",
    lifespan: "Up to 10 years",
    description: "A striking hummingbird with a vibrant green and violet plumage, commonly found in the highlands of Central America."
  },
  "071.Long_tailed_Jaeger": {
    id: "071",
    name: "Long-tailed Jaeger",
    scientificName: "Stercorarius longicaudus",
    image: "https://images.unsplash.com/photo-1582074717133-2da4f26b55b4?auto=format&fit=crop&q=80",
    habitat: "Arctic regions and coastal areas",
    diet: "Fish, small birds, and eggs",
    lifespan: "Up to 10 years",
    description: "A large seabird with long, pointed tail feathers, found in the Arctic and known for its aggressive hunting behavior."
  },
  "072.Pomarine_Jaeger": {
    id: "072",
    name: "Pomarine Jaeger",
    scientificName: "Stercorarius pomarinus",
    image: "https://images.unsplash.com/photo-1573508587415-6a828e5308cc?auto=format&fit=crop&q=80",
    habitat: "Coastal and open ocean regions",
    diet: "Fish, small birds, and scavenged food",
    lifespan: "Up to 15 years",
    description: "A robust seabird with a distinctive pom-pom like tail, often seen harassing other birds for food."
  },
  "073.Blue_Jay": {
    id: "073",
    name: "Blue Jay",
    scientificName: "Cyanocitta cristata",
    image: "https://images.unsplash.com/photo-1601669046544-b498e3e56a99?auto=format&fit=crop&q=80",
    habitat: "Woodlands, gardens, and parks in North America",
    diet: "Nuts, seeds, fruits, and insects",
    lifespan: "Up to 26 years",
    description: "A striking bird with blue, white, and black plumage, known for its intelligence and complex vocalizations."
  },
  "074.Florida_Jay": {
    id: "074",
    name: "Florida Jay",
    scientificName: "Cyanocitta cristata florincola",
    image: "https://images.unsplash.com/photo-1616163156761-e4f22b0ecb64?auto=format&fit=crop&q=80",
    habitat: "Florida pine forests and oak groves",
    diet: "Acorns, seeds, fruits, and small insects",
    lifespan: "Up to 15 years",
    description: "A subspecies of the Blue Jay, with similar characteristics but found specifically in the Florida region."
  },
  "075.Green_Jay": {
    id: "075",
    name: "Green Jay",
    scientificName: "Cyanocorax yncas",
    image: "https://images.unsplash.com/photo-1586280175611-7d2a3b96e9fe?auto=format&fit=crop&q=80",
    habitat: "Woodlands and scrublands in southern Texas and Mexico",
    diet: "Fruits, seeds, and insects",
    lifespan: "Up to 15 years",
    description: "A colorful jay with a vibrant green body and a striking blue head, found in the southwestern U.S. and Mexico."
  },
  "076.Dark_eyed_Junco": {
    id: "076",
    name: "Dark-eyed Junco",
    scientificName: "Junco hyemalis",
    image: "https://images.unsplash.com/photo-1584585203426-77bfc687ce18?auto=format&fit=crop&q=80",
    habitat: "Forests, gardens, and mountainous areas across North America",
    diet: "Seeds, insects, and small fruits",
    lifespan: "Up to 11 years",
    description: "A small sparrow with a dark-colored head and chest, commonly found in North American winter environments."
  },
  "077.Tropical_Kingbird": {
    id: "077",
    name: "Tropical Kingbird",
    scientificName: "Tyrannus melancholicus",
    image: "https://images.unsplash.com/photo-1580485478762-2b0a8f9d3f77?auto=format&fit=crop&q=80",
    habitat: "Tropical and subtropical regions of Central and South America",
    diet: "Insects, small fruit, and nectar",
    lifespan: "Up to 7 years",
    description: "A large flycatcher with a bold yellow belly and gray wings, known for its aggressive defense of territory."
  },
  "078.Gray_Kingbird": {
    id: "078",
    name: "Gray Kingbird",
    scientificName: "Tyrannus dominicensis",
    image: "https://images.unsplash.com/photo-1597396262851-7d1b19a3b707?auto=format&fit=crop&q=80",
    habitat: "Tropical forests, mangroves, and coastal regions",
    diet: "Insects, small lizards, and fruit",
    lifespan: "Up to 5 years",
    description: "A gray and white flycatcher with a distinctive black mask, found in tropical areas of the Caribbean."
  },
  "079.Belted_Kingfisher": {
    id: "079",
    name: "Belted Kingfisher",
    scientificName: "Megaceryle alcyon",
    image: "https://images.unsplash.com/photo-1603560990986-61ff2d49ecb5?auto=format&fit=crop&q=80",
    habitat: "Rivers, lakes, and coastal areas in North America",
    diet: "Fish, crustaceans, and aquatic insects",
    lifespan: "Up to 10 years",
    description: "A medium-sized kingfisher with a striking blue and white plumage, known for its skill in fishing."
  },
  "080.Green_Kingfisher": {
    id: "080",
    name: "Green Kingfisher",
    scientificName: "Chloroceryle americana",
    image: "https://images.unsplash.com/photo-1602271475149-f21618e5e179?auto=format&fit=crop&q=80",
    habitat: "Rivers, lakes, and wetlands in Central America",
    diet: "Fish, insects, and crustaceans",
    lifespan: "Up to 10 years",
    description: "A small kingfisher with a vivid green back and a white belly, found near freshwater sources in Central America."
  },
  "081.Pied_Kingfisher": {
    id: "081",
    name: "Pied Kingfisher",
    scientificName: "Ceryle rudis",
    image: "https://images.unsplash.com/photo-1582069360612-b1e2f2760204?auto=format&fit=crop&q=80",
    habitat: "Rivers, lakes, and wetlands across Africa and Asia",
    diet: "Fish and small aquatic invertebrates",
    lifespan: "Up to 10 years",
    description: "A distinctive black-and-white kingfisher with a crest, known for its hovering flight while hunting."
  },
  "082.Ringed_Kingfisher": {
    id: "082",
    name: "Ringed Kingfisher",
    scientificName: "Ceryle torquata",
    image: "https://images.unsplash.com/photo-1560770166-9d3fd05eaa4c?auto=format&fit=crop&q=80",
    habitat: "Rivers, lakes, and coastal regions in Central and South America",
    diet: "Fish, crustaceans, and insects",
    lifespan: "Up to 9 years",
    description: "A large kingfisher with a distinctive white ring around its neck and a loud, harsh call."
  },
  "083.White_breasted_Kingfisher": {
    id: "083",
    name: "White-breasted Kingfisher",
    scientificName: "Halcyon smyrnensis",
    image: "https://images.unsplash.com/photo-1571771606791-d490f207c6c4?auto=format&fit=crop&q=80",
    habitat: "Woodlands, gardens, and agricultural areas in Asia and parts of Africa",
    diet: "Fish, small reptiles, and insects",
    lifespan: "Up to 8 years",
    description: "A brightly colored kingfisher with a blue back and white underparts, often found near water."
  },
  "084.Red_legged_Kittiwake": {
    id: "084",
    name: "Red-legged Kittiwake",
    scientificName: "Rissa brevirostris",
    image: "https://images.unsplash.com/photo-1585235390590-4b4424de4a29?auto=format&fit=crop&q=80",
    habitat: "Coastal regions of Alaska and the Arctic",
    diet: "Fish, crustaceans, and mollusks",
    lifespan: "Up to 20 years",
    description: "A seabird with red legs and a white body, often seen in the Arctic during the summer months."
  },
  "085.Horned_Lark": {
    id: "085",
    name: "Horned Lark",
    scientificName: "Eremophila alpestris",
    image: "https://images.unsplash.com/photo-1602584997053-0f08a1a8e6d2?auto=format&fit=crop&q=80",
    habitat: "Open fields, grasslands, and deserts across North America and Eurasia",
    diet: "Seeds, insects, and small plants",
    lifespan: "Up to 9 years",
    description: "A small, ground-dwelling bird with distinctive black horns on its head, known for its melodious song."
  },
  "086.Pacific_Loon": {
    id: "086",
    name: "Pacific Loon",
    scientificName: "Gavia pacifica",
    image: "https://images.unsplash.com/photo-1583548242458-bfa63f5ed635?auto=format&fit=crop&q=80",
    habitat: "Coastal and inland waters in North America and East Asia",
    diet: "Fish and aquatic invertebrates",
    lifespan: "Up to 15 years",
    description: "A medium-sized waterbird with a striking black and white plumage, known for its eerie calls."
  },
  "087.Mallard": {
    id: "087",
    name: "Mallard",
    scientificName: "Anas platyrhynchos",
    image: "https://images.unsplash.com/photo-1566785739-5f70fa5ebf3a?auto=format&fit=crop&q=80",
    habitat: "Freshwater lakes, rivers, and marshes across North America, Europe, and Asia",
    diet: "Aquatic plants, seeds, insects, and small fish",
    lifespan: "Up to 10 years",
    description: "A common duck species, with the male having an iridescent green head and a characteristic quack."
  },
  "088.Western_Meadowlark": {
    id: "088",
    name: "Western Meadowlark",
    scientificName: "Sturnella neglecta",
    image: "https://images.unsplash.com/photo-1576277594917-eccab5006600?auto=format&fit=crop&q=80",
    habitat: "Grasslands and prairies in North America",
    diet: "Insects, seeds, and berries",
    lifespan: "Up to 12 years",
    description: "A songbird known for its bright yellow chest and melodious song, often seen in open fields."
  },
  "089.Hooded_Merganser": {
    id: "089",
    name: "Hooded Merganser",
    scientificName: "Lophodytes cucullatus",
    image: "https://images.unsplash.com/photo-1573078497062-746e15fcfbf0?auto=format&fit=crop&q=80",
    habitat: "Freshwater lakes, rivers, and ponds across North America",
    diet: "Fish, aquatic invertebrates, and small crustaceans",
    lifespan: "Up to 10 years",
    description: "A striking diving duck with a large, fan-shaped crest, known for its fish-catching abilities."
  },
  "090.Red_breasted_Merganser": {
    id: "090",
    name: "Red-breasted Merganser",
    scientificName: "Mergus serrator",
    image: "https://images.unsplash.com/photo-1579274871601-9b40093a86d9?auto=format&fit=crop&q=80",
    habitat: "Coastal waters, rivers, and lakes across North America and Europe",
    diet: "Fish and aquatic invertebrates",
    lifespan: "Up to 12 years",
    description: "A slender diving duck with a distinctive red breast and sharp, serrated bill, ideal for catching fish."
  },
  "091.Mockingbird": {
    id: "091",
    name: "Mockingbird",
    scientificName: "Mimus polyglottos",
    image: "https://images.unsplash.com/photo-1580414693857-96f7c212a295?auto=format&fit=crop&q=80",
    habitat: "Urban areas, forests, and suburban regions across North America",
    diet: "Insects, berries, and small fruits",
    lifespan: "Up to 8 years",
    description: "A versatile bird known for its ability to mimic the sounds of other birds and even mechanical noises."
  },
  "092.Nighthawk": {
    id: "092",
    name: "Nighthawk",
    scientificName: "Chordeiles minor",
    image: "https://images.unsplash.com/photo-1567521517-e6d8bc925bb7?auto=format&fit=crop&q=80",
    habitat: "Open areas, deserts, and forests across North America",
    diet: "Insects, particularly moths and beetles",
    lifespan: "Up to 5 years",
    description: "A nocturnal bird known for its distinctive flight pattern and haunting call during the evening hours."
  },
  "093.Clark_Nutcracker": {
    id: "093",
    name: "Clark's Nutcracker",
    scientificName: "Nucifraga columbiana",
    image: "https://images.unsplash.com/photo-1565444265-0a2051fc7be3?auto=format&fit=crop&q=80",
    habitat: "Mountain forests in the western United States",
    diet: "Pine seeds, insects, and berries",
    lifespan: "Up to 15 years",
    description: "A crow-like bird with a long, pointed bill, known for its ability to store pine seeds for later consumption."
  },
  "094.White_breasted_Nuthatch": {
    id: "094",
    name: "White-breasted Nuthatch",
    scientificName: "Sitta carolinensis",
    image: "https://images.unsplash.com/photo-1564422215-2f67a91c0d8c?auto=format&fit=crop&q=80",
    habitat: "Woodlands and forests across North America",
    diet: "Insects, seeds, and nuts",
    lifespan: "Up to 9 years",
    description: "A small, acrobatic bird known for its habit of climbing trees headfirst while foraging for food."
  },
  "095.Baltimore_Oriole": {
    id: "095",
    name: "Baltimore Oriole",
    scientificName: "Icterus galbula",
    image: "https://images.unsplash.com/photo-1573247021389-c8e52ff975d1?auto=format&fit=crop&q=80",
    habitat: "Woodlands, forests, and suburban areas in eastern North America",
    diet: "Nectar, fruits, and insects",
    lifespan: "Up to 10 years",
    description: "A vibrant orange and black bird known for its distinctive call and preference for hanging from branches while feeding."
  },
  "096.Hooded_Oriole": {
    id: "096",
    name: "Hooded Oriole",
    scientificName: "Icterus cucullatus",
    image: "https://images.unsplash.com/photo-1571787542831-255c459522be?auto=format&fit=crop&q=80",
    habitat: "Deserts, woodlands, and suburban areas in the southwestern United States",
    diet: "Nectar, fruits, and insects",
    lifespan: "Up to 12 years",
    description: "A striking oriole with a bright yellow and black plumage, known for its large, hanging nests."
  },
  "097.Orchard_Oriole": {
    id: "097",
    name: "Orchard Oriole",
    scientificName: "Icterus spurius",
    image: "https://images.unsplash.com/photo-1560425402-90ab8a65f34b?auto=format&fit=crop&q=80",
    habitat: "Woodlands, orchards, and farmlands across North America",
    diet: "Fruits, insects, and nectar",
    lifespan: "Up to 8 years",
    description: "A smaller oriole species, with a dark red breast and a preference for fruit-laden trees."
  },
  "098.Scott_Oriole": {
    id: "098",
    name: "Scott's Oriole",
    scientificName: "Icterus parisorum",
    image: "https://images.unsplash.com/photo-1574312654788-c6051a653978?auto=format&fit=crop&q=80",
    habitat: "Desert areas, particularly in the southwestern United States",
    diet: "Nectar, fruits, and insects",
    lifespan: "Up to 7 years",
    description: "A yellow and black oriole, often found in arid environments, known for its loud, clear whistle."
  },
  "099.Ovenbird": {
    id: "099",
    name: "Ovenbird",
    scientificName: "Seiurus aurocapilla",
    image: "https://images.unsplash.com/photo-1578596133282-27456016a243?auto=format&fit=crop&q=80",
    habitat: "Woodlands and forests across North America",
    diet: "Insects, spiders, and other small invertebrates",
    lifespan: "Up to 6 years",
    description: "A small, olive-colored bird known for its distinctive 'oven-shaped' nests built on the forest floor."
  },
  "100.Brown_Pelican": {
    id: "100",
    name: "Brown Pelican",
    scientificName: "Pelecanus occidentalis",
    image: "https://images.unsplash.com/photo-1602972634481-dfc2d8e69bc8?auto=format&fit=crop&q=80",
    habitat: "Coastal areas and estuaries across the Americas",
    diet: "Fish and marine invertebrates",
    lifespan: "Up to 16 years",
    description: "A large, coastal bird with a distinctive long bill and a remarkable fishing technique that involves diving into the water."
  } 
  };
  