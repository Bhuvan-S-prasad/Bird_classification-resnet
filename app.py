import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms, models
from PIL import Image

# Flask setup
app = Flask(__name__,  static_folder='static')
app.config['UPLOAD_FOLDER'] = './static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
class_names = [
    "001.Black_footed_Albatross", "002.Laysan_Albatross", "003.Sooty_Albatross", 
    "004.Groove_billed_Ani", "005.Crested_Auklet", "006.Least_Auklet", 
    "007.Parakeet_Auklet", "008.Rhinoceros_Auklet", "009.Brewer_Blackbird", 
    "010.Red_winged_Blackbird", "011.Rusty_Blackbird", "012.Yellow_headed_Blackbird", 
    "013.Bobolink", "014.Indigo_Bunting", "015.Lazuli_Bunting", "016.Painted_Bunting", 
    "017.Cardinal", "018.Spotted_Catbird", "019.Gray_Catbird", 
    "020.Yellow_breasted_Chat", "021.Eastern_Towhee", "022.Chuck_will_Widow", 
    "023.Brandt_Cormorant", "024.Red_faced_Cormorant", "025.Pelagic_Cormorant", 
    "026.Bronzed_Cowbird", "027.Shiny_Cowbird", "028.Brown_Creeper", 
    "029.American_Crow", "030.Fish_Crow", "031.Black_billed_Cuckoo", 
    "032.Mangrove_Cuckoo", "033.Yellow_billed_Cuckoo", "034.Gray_crowned_Rosy_Finch", 
    "035.Purple_Finch", "036.Northern_Flicker", "037.Acadian_Flycatcher", 
    "038.Great_Crested_Flycatcher", "039.Least_Flycatcher", "040.Olive_sided_Flycatcher", 
    "041.Scissor_tailed_Flycatcher", "042.Vermilion_Flycatcher", 
    "043.Yellow_bellied_Flycatcher", "044.Frigatebird", "045.Northern_Fulmar", 
    "046.Gadwall", "047.American_Goldfinch", "048.European_Goldfinch", 
    "049.Boat_tailed_Grackle", "050.Eared_Grebe", "051.Horned_Grebe", 
    "052.Pied_billed_Grebe", "053.Western_Grebe", "054.Blue_Grosbeak", 
    "055.Evening_Grosbeak", "056.Pine_Grosbeak", "057.Rose_breasted_Grosbeak", 
    "058.Pigeon_Guillemot", "059.California_Gull", "060.Glaucous_winged_Gull", 
    "061.Heermann_Gull", "062.Herring_Gull", "063.Ivory_Gull", 
    "064.Ring_billed_Gull", "065.Slaty_backed_Gull", "066.Western_Gull", 
    "067.Anna_Hummingbird", "068.Ruby_throated_Hummingbird", 
    "069.Rufous_Hummingbird", "070.Green_Violetear", "071.Long_tailed_Jaeger", 
    "072.Pomarine_Jaeger", "073.Blue_Jay", "074.Florida_Jay", "075.Green_Jay", 
    "076.Dark_eyed_Junco", "077.Tropical_Kingbird", "078.Gray_Kingbird", 
    "079.Belted_Kingfisher", "080.Green_Kingfisher", "081.Pied_Kingfisher", 
    "082.Ringed_Kingfisher", "083.White_breasted_Kingfisher", 
    "084.Red_legged_Kittiwake", "085.Horned_Lark", "086.Pacific_Loon", 
    "087.Mallard", "088.Western_Meadowlark", "089.Hooded_Merganser", 
    "090.Red_breasted_Merganser", "091.Mockingbird", "092.Nighthawk", 
    "093.Clark_Nutcracker", "094.White_breasted_Nuthatch", 
    "095.Baltimore_Oriole", "096.Hooded_Oriole", "097.Orchard_Oriole", 
    "098.Scott_Oriole", "099.Ovenbird", "100.Brown_Pelican", "unknown"
]



bird_details = {
    "001.Black_footed_Albatross": {
      "id": "001",
      "name": "Black-footed Albatross",
      "scientificName": "Phoebastria nigripes",
      "habitat": "Open ocean of the North Pacific; nests on remote islands",
      "diet": "Squid, fish, and fish eggs. Often follows fishing vessels for discarded fish",
      "lifespan": "12-40 years",
      "description": "A large seabird with dark plumage and distinctive black feet. Known for their incredible ability to glide over ocean waves for hours without flapping their wings."
    },
    "002.Laysan_Albatross": {
      "id": "002",
      "name": "Laysan Albatross",
      "scientificName": "Phoebastria immutabilis",
      "habitat": "Open ocean of the North Pacific; breeds primarily on Hawaiian islands",
      "diet": "Squid, fish, and crustaceans",
      "lifespan": "Up to 40 years",
      "description": "Distinguished by its white body and dark wings, known for their elaborate courtship dances and strong pair bonds that can last for life."
    },
    "003.Sooty_Albatross": {
    "id": "003",
    "name": "Sooty Albatross",
    "scientificName": "Phoebetria fusca",
    "habitat": "Southern oceans, particularly around sub-Antarctic islands",
    "diet": "Squid, fish, and marine crustaceans",
    "lifespan": "Up to 35 years",
    "description": "Dark-plumaged seabird with distinctive white eye-rings and yellow stripe on the bill. Expert glider in strong winds."
  },
  "004.Groove_billed_Ani": {
    "id": "004",
    "name": "Groove-billed Ani",
    "scientificName": "Crotophaga sulcirostris",
    "habitat": "Open and semi-open areas, farmland, and scrubby woodland",
    "diet": "Insects, lizards, and fruits",
    "lifespan": "4-8 years",
    "description": "Social bird with distinctive grooved bill, often seen in small groups. Known for cooperative breeding where several females lay eggs in one nest."
  },
  "005.Crested_Auklet": {
    "id": "005",
    "name": "Crested Auklet",
    "scientificName": "Aethia cristatella",
    "habitat": "North Pacific Ocean and Bering Sea",
    "diet": "Plankton and small marine invertebrates",
    "lifespan": "Up to 20 years",
    "description": "Distinguished by its forward-curving crest and orange bill. Known for their citrus-like scent during breeding season."
  },
  "006.Least_Auklet": {
    "id": "006",
    "name": "Least Auklet",
    "scientificName": "Aethia pusilla",
    "habitat": "Bering Sea and North Pacific Ocean",
    "diet": "Small marine crustaceans and plankton",
    "lifespan": "Up to 12 years",
    "description": "Smallest diving bird in North America. Recognizable by their small size and white facial plumes during breeding season."
  },
  "007.Parakeet_Auklet": {
    "id": "007",
    "name": "Parakeet Auklet",
    "scientificName": "Aethia psittacula",
    "habitat": "North Pacific Ocean and Bering Sea",
    "diet": "Small fish, squid, and marine invertebrates",
    "lifespan": "15-20 years",
    "description": "Medium-sized seabird with distinctive upturned bill and white facial plumes. Excellent underwater swimmer."
  },
  "008.Rhinoceros_Auklet": {
    "id": "008",
    "name": "Rhinoceros Auklet",
    "scientificName": "Cerorhinca monocerata",
    "habitat": "Coastal waters of the North Pacific",
    "diet": "Small fish and squid",
    "lifespan": "Up to 30 years",
    "description": "Named for the horn-like projection on its bill during breeding season. Nocturnal on breeding colonies."
  },
  "009.Brewer_Blackbird": {
    "id": "009",
    "name": "Brewer's Blackbird",
    "scientificName": "Euphagus cyanocephalus",
    "habitat": "Open and semi-open areas, urban parks, and farmland",
    "diet": "Insects, seeds, and grains",
    "lifespan": "12-15 years",
    "description": "Glossy black bird with piercing yellow eyes. Highly adaptable to urban environments."
  },
  "010.Red_winged_Blackbird": {
    "id": "010",
    "name": "Red-winged Blackbird",
    "scientificName": "Agelaius phoeniceus",
    "habitat": "Marshes, wetlands, and grasslands",
    "diet": "Insects, seeds, and grains",
    "lifespan": "15-20 years",
    "description": "Males are black with distinctive red and yellow shoulder patches. One of the most abundant birds in North America."
  },
  "011.Rusty_Blackbird": {
    "id": "011",
    "name": "Rusty Blackbird",
    "scientificName": "E    uphagus carolinus",
    "habitat": "Boreal forests, wetlands, and marshes",
    "diet": "Insects, seeds, and small aquatic invertebrates",
    "lifespan": "8-12 years",
    "description": "Medium-sized blackbird with rusty-brown plumage in winter. Known for their sharp, creaky song."
  },
  "012.Yellow_headed_Blackbird": {
    "id": "012",
    "name": "Yellow-headed Blackbird",
    "scientificName": "Xanthocephalus xanthocephalus",
    "habitat": "Freshwater marshes and wetlands",
    "diet": "Insects and seeds",
    "lifespan": "10-15 years",
    "description": "Striking bird with bright yellow head and chest. Found in large flocks during migration."
  },
  "013.Bobolink": {
    "id": "013",
    "name": "Bobolink",
    "scientificName": "Dolichonyx oryzivorus",
    "habitat": "Grasslands and prairies",
    "diet": "Seeds, grains, and insects",
    "lifespan": "4-9 years",
    "description": "Known for their bubbly, metallic song. Males have distinctive black and white plumage during the breeding season."
  },
  "014.Indigo_Bunting": {
    "id": "014",
    "name": "Indigo Bunting",
    "scientificName": "Passerina cyanea",
    "habitat": "Open woodlands, farmlands, and forest edges",
    "diet": "Insects, seeds, and berries",
    "lifespan": "5-10 years",
    "description": "Brilliant blue male plumage during summer. Females are brownish and well-camouflaged."
  },
  "015.Lazuli_Bunting": {
    "id": "015",
    "name": "Lazuli Bunting",
    "scientificName": "Passerina amoena",
    "habitat": "Shrubby areas, open woodlands, and riparian zones",
    "diet": "Seeds, insects, and fruits",
    "lifespan": "3-8 years",
    "description": "Male has vivid blue plumage with an orange breast. Known for their melodious song."
  },
  "016.Painted_Bunting": {
    "id": "016",
    "name": "Painted Bunting",
    "scientificName": "Passerina ciris",
    "habitat": "Woodland edges, thickets, and brushy areas",
    "diet": "Seeds and insects",
    "lifespan": "5-8 years",
    "description": "One of the most colorful songbirds in North America. Males are a vivid blend of blue, green, and red."
  },
  "017.Cardinal": {
    "id": "017",
    "name": "Northern Cardinal",
    "scientificName": "Cardinalis cardinalis",
    "habitat": "Woodlands, gardens, and shrublands",
    "diet": "Seeds, fruits, and insects",
    "lifespan": "3-15 years",
    "description": "Bright red males are a familiar sight in backyards. Females are tan with reddish tinges."
  },
  "018.Spotted_Catbird": {
    "id": "018",
    "name": "Spotted Catbird",
    "scientificName": "Ailuroedus maculosus",
    "habitat": "Rainforests and dense woodlands",
    "diet": "Fruits and insects",
    "lifespan": "Up to 10 years",
    "description": "Named for their cat-like calls. Green plumage with spots gives them a unique appearance."
  },
  "019.Gray_Catbird": {
    "id": "019",
    "name": "Gray Catbird",
    "scientificName": "Dumetella carolinensis",
    "habitat": "Thickets, forests, and suburban areas",
    "diet": "Insects, fruits, and berries",
    "lifespan": "4-10 years",
    "description": "Sleek gray bird with a distinctive black cap. Known for their varied song and mimicry."
  },
  "020.Yellow_breasted_Chat": {
    "id": "020",
    "name": "Yellow-breasted Chat",
    "scientificName": "Icteria virens",
    "habitat": "Thickets, shrubs, and forest edges",
    "diet": "Insects, fruits, and berries",
    "lifespan": "Up to 10 years",
    "description": "Largest warbler in North America, with a bright yellow breast and bold behavior."
  },
  "021.Eastern_Towhee": {
    "id": "021",
    "name": "Eastern Towhee",
    "scientificName": "Pipilo erythrophthalmus",
    "habitat": "Shrubby areas, woodlands, and forest edges",
    "diet": "Seeds, fruits, and insects",
    "lifespan": "4-9 years",
    "description": "Distinctive black, white, and rufous plumage. Known for their sharp 'drink your tea' call."
  },
  "022.Chuck_will_Widow": {
      "id": "022",
      "name": "Chuck-will's-widow",
      "scientificName": "Antrostomus carolinensis",
      "habitat": "Open woodlands and forest edges",
      "diet": "Insects, especially moths and beetles",
      "lifespan": "Up to 5 years",
      "description": "A nocturnal bird with long wings and a distinctive churring call. Related to the whip-poor-will."
    },
    "023.Brandt_Cormorant": {
      "id": "023",
      "name": "Brandt's Cormorant",
      "scientificName": "Phalacrocorax penicillatus",
      "habitat": "Coastal cliffs and rocky shores",
      "diet": "Fish, crustaceans, and invertebrates",
      "lifespan": "Up to 13 years",
      "description": "A large, dark seabird with a striking blue throat and long, pointed bill."
    },
    "024.Red_faced_Cormorant": {
      "id": "024",
      "name": "Red-faced Cormorant",
      "scientificName": "Phalacrocorax urile",
      "habitat": "Rocky coastlines and islands",
      "diet": "Fish, squid, and other marine life",
      "lifespan": "Up to 8 years",
      "description": "A striking seabird with a bright red face and sharp bill, known for diving to catch prey."
    },
    "025.Pelagic_Cormorant": {
      "id": "025",
      "name": "Pelagic Cormorant",
      "scientificName": "Phalacrocorax pelagicus",
      "habitat": "Coastal waters and rocky islands",
      "diet": "Fish and invertebrates",
      "lifespan": "Up to 12 years",
      "description": "A small, dark seabird with a slender body and striking greenish-black plumage."
    },
    "026.Bronzed_Cowbird": {
      "id": "026",
      "name": "Bronzed Cowbird",
      "scientificName": "Molothrus aeneus",
      "habitat": "Open grasslands and agricultural areas",
      "diet": "Seeds, grains, and insects",
      "lifespan": "Up to 8 years",
      "description": "A shiny, dark-colored bird often seen in large flocks, known for parasitizing other birds' nests."
    },
    "027.Shiny_Cowbird": {
      "id": "027",
      "name": "Shiny Cowbird",
      "scientificName": "Molothrus bonariensis",
      "habitat": "Open fields, pastures, and agricultural areas",
      "diet": "Insects, seeds, and grains",
      "lifespan": "Up to 8 years",
      "description": "A glossy, black bird with a short bill, often seen near livestock or human settlements."
    },
    "028.Brown_Creeper": {
      "id": "028",
      "name": "Brown Creeper",
      "scientificName": "Certhia americana",
      "habitat": "Forest habitats, especially coniferous trees",
      "diet": "Insects and spiders found on tree bark",
      "lifespan": "Up to 7 years",
      "description": "A small, cryptic bird with an upward-curved bill, often seen creeping along tree trunks."
    },
    "029.American_Crow": {
      "id": "029",
      "name": "American Crow",
      "scientificName": "Corvus brachyrhynchos",
      "habitat": "A wide range of habitats, from forests to urban areas",
      "diet": "Omnivorous, including insects, fruits, and small animals",
      "lifespan": "Up to 8 years",
      "description": "A highly intelligent bird known for its black feathers and loud, harsh call. Often seen in large groups."
    },
    "030.Fish_Crow": {
      "id": "030",
      "name": "Fish Crow",
      "scientificName": "Corvus ossifragus",
      "habitat": "Coastal areas, wetlands, and estuaries",
      "diet": "Fish, crabs, and other small aquatic animals",
      "lifespan": "Up to 7 years",
      "description": "A smaller, more specialized cousin of the American Crow, known for its preference for coastal habitats."
    },
  "031.Black_billed_Cuckoo": {
    "id": "031",
    "name": "Black-billed Cuckoo",
    "scientificName": "Coccyzus erythropthalmus",
    "habitat": "Deciduous forests, woodlands, and suburban areas",
    "diet": "Insects, especially caterpillars",
    "lifespan": "Up to 10 years",
    "description": "A medium-sized bird known for its black bill and distinctive, cuckoo-like call."
  },
  "032.Mangrove_Cuckoo": {
    "id": "032",
    "name": "Mangrove Cuckoo",
    "scientificName": "Coccyzus minor",
    "habitat": "Mangrove forests and coastal areas",
    "diet": "Insects and small vertebrates",
    "lifespan": "Up to 8 years",
    "description": "A secretive, large cuckoo with a long tail and distinct call, often found in coastal mangroves."
  },
  "033.Yellow_billed_Cuckoo": {
    "id": "033",
    "name": "Yellow-billed Cuckoo",
    "scientificName": "Coccyzus americanus",
    "habitat": "Woodlands, forests, and riparian areas",
    "diet": "Insects, particularly caterpillars",
    "lifespan": "Up to 12 years",
    "description": "A bird with a bright yellow bill, known for its distinct call and insectivorous diet."
  },
  "034.Gray_crowned_Rosy_Finch": {
    "id": "034",
    "name": "Gray-crowned Rosy Finch",
    "scientificName": "Leucosticte tephrocotis",
    "habitat": "Alpine and mountainous regions",
    "diet": "Seeds, insects, and berries",
    "lifespan": "Up to 6 years",
    "description": "A small, high-altitude finch with a gray crown and rosy pink plumage."
  },
  "035.Purple_Finch": {
    "id": "035",
    "name": "Purple Finch",
    "scientificName": "Haemorhous purpureus",
    "habitat": "Woodlands, coniferous forests, and suburbs",
    "diet": "Seeds, fruits, and insects",
    "lifespan": "Up to 7 years",
    "description": "A medium-sized finch with deep red plumage and a distinctive song."
  },
  "036.Northern_Flicker": {
    "id": "036",
    "name": "Northern Flicker",
    "scientificName": "Colaptes auratus",
    "habitat": "Wooded areas, including forests and urban parks",
    "diet": "Ants, beetles, and other insects",
    "lifespan": "Up to 9 years",
    "description": "A large woodpecker known for its distinctive call and brownish plumage with black spots."
  },
  "037.Acadian_Flycatcher": {
    "id": "037",
    "name": "Acadian Flycatcher",
    "scientificName": "Empidonax virescens",
    "habitat": "Wooded areas, especially near streams",
    "diet": "Insects, especially flying insects",
    "lifespan": "Up to 5 years",
    "description": "A small flycatcher with olive-green plumage and a distinctive 'fitz-bew' call."
  },
  "038.Great_Crested_Flycatcher": {
    "id": "038",
    "name": "Great Crested Flycatcher",
    "scientificName": "Myiarchus crinitus",
    "habitat": "Woodlands, forests, and suburban areas",
    "diet": "Insects, berries, and small fruits",
    "lifespan": "Up to 10 years",
    "description": "A medium-sized flycatcher with a large crest and a yellow belly."
  },
  "039.Least_Flycatcher": {
    "id": "039",
    "name": "Least Flycatcher",
    "scientificName": "Empidonax minimus",
    "habitat": "Wooded areas, particularly near streams",
    "diet": "Insects, especially small flying insects",
    "lifespan": "Up to 7 years",
    "description": "A small, olive-green flycatcher known for its sharp 'che-bek' call."
  },
  "040.Olive_sided_Flycatcher": {
    "id": "040",
    "name": "Olive-sided Flycatcher",
    "scientificName": "Contopus cooperi",
    "habitat": "Forests, especially in the western U.S.",
    "diet": "Insects, particularly large flying insects",
    "lifespan": "Up to 8 years",
    "description": "A medium-sized flycatcher with olive-green plumage and a distinctive 'quick, three beers' call."
  },
  "041.Scissor_tailed_Flycatcher": {
    "id": "041",
    "name": "Scissor-tailed Flycatcher",
    "scientificName": "Tyrannus forficatus",
    "habitat": "Open woodlands and prairies",
    "diet": "Insects, especially flying insects",
    "lifespan": "Up to 6 years",
    "description": "A striking flycatcher with long, scissor-like tail feathers and a pale underbelly."
  },
  "042.Vermilion_Flycatcher": {
    "id": "042",
    "name": "Vermilion Flycatcher",
    "scientificName": "Pyrocephalus rubinus",
    "habitat": "Open woodlands, riparian areas, and deserts",
    "diet": "Insects, especially beetles and ants",
    "lifespan": "Up to 5 years",
    "description": "A small, striking flycatcher with brilliant vermilion plumage and dark wings."
  },
  "043.Yellow_bellied_Flycatcher": {
    "id": "043",
    "name": "Yellow-bellied Flycatcher",
    "scientificName": "Empidonax flaviventris",
    "habitat": "Wooded areas, especially near water",
    "diet": "Insects, especially moths and beetles",
    "lifespan": "Up to 5 years",
    "description": "A small flycatcher with a yellow belly and olive-green plumage, known for its soft 'che-bek' call."
  },
  "044.Frigatebird": {
    "id": "044",
    "name": "Frigatebird",
    "scientificName": "Fregata magnificens",
    "habitat": "Tropical coastal regions and islands",
    "diet": "Fish, squid, and other marine life",
    "lifespan": "Up to 15 years",
    "description": "A large seabird with a forked tail and a distinctive, red throat pouch."
  },
  "045.Northern_Fulmar": {
    "id": "045",
    "name": "Northern Fulmar",
    "scientificName": "Fulmarus glacialis",
    "habitat": "Coastal cliffs and tundra",
    "diet": "Fish, squid, and krill",
    "lifespan": "Up to 40 years",
    "description": "A seabird with a heavy bill, known for its ability to glide over the ocean for long distances."
  },
  "046.Gadwall": {
    "id": "046",
    "name": "Gadwall",
    "scientificName": "Mareca strepera",
    "habitat": "Wetlands, lakes, and rivers",
    "diet": "Aquatic plants, seeds, and small invertebrates",
    "lifespan": "Up to 10 years",
    "description": "A medium-sized dabbling duck with a distinctive grayish-brown plumage and a white belly."
  },
  "047.American_Goldfinch": {
    "id": "047",
    "name": "American Goldfinch",
    "scientificName": "Spinus tristis",
    "habitat": "Fields, gardens, and suburban areas",
    "diet": "Seeds, especially sunflower seeds",
    "lifespan": "Up to 11 years",
    "description": "A small finch with bright yellow plumage in males, known for its cheerful, twitting call."
  },
  "048.European_Goldfinch": {
    "id": "048",
    "name": "European Goldfinch",
    "scientificName": "Carduelis carduelis",
    "habitat": "Woodlands, gardens, and farmland",
    "diet": "Seeds, particularly thistle and dandelion seeds",
    "lifespan": "Up to 9 years",
    "description": "A colorful finch with a red face, yellow wing bars, and a distinctive trilling song."
  },
  "049.Boat_tailed_Grackle": {
    "id": "049",
    "name": "Boat-tailed Grackle",
    "scientificName": "Quiscalus major",
    "habitat": "Coastal marshes, salt flats, and wetlands",
    "diet": "Insects, seeds, and small vertebrates",
    "lifespan": "Up to 9 years",
    "description": "A large blackbird with an iridescent body and a long, boat-shaped tail."
  },
  "050.Eared_Grebe": {
    "id": "050",
    "name": "Eared Grebe",
    "scientificName": "Podiceps nigricollis",
    "habitat": "Lakes and marshes",
    "diet": "Small fish, aquatic insects, and crustaceans",
    "lifespan": "Up to 5 years",
    "description": "A small waterbird with dark plumage and distinctive, ear-like tufts during the breeding season."
  },
  "051.Horned_Grebe": {
    "id": "051",
    "name": "Horned Grebe",
    "scientificName": "Podiceps auritus",
    "habitat": "Lakes, ponds, and marshes",
    "diet": "Small fish, aquatic insects, and crustaceans",
    "lifespan": "Up to 5 years",
    "description": "A small waterbird with a distinctive black and white plumage and tufted horns during the breeding season."
  },
  "052.Pied_billed_Grebe": {
    "id": "052",
    "name": "Pied-billed Grebe",
    "scientificName": "Podilymbus podiceps",
    "habitat": "Lakes, ponds, and marshes",
    "diet": "Small fish, amphibians, and aquatic invertebrates",
    "lifespan": "Up to 10 years",
    "description": "A small grebe with a distinctive black ring around its bill and a chunky, rounded body."
  },
  "053.Western_Grebe": {
    "id": "053",
    "name": "Western Grebe",
    "scientificName": "Aechmophorus occidentalis",
    "habitat": "Large lakes and reservoirs",
    "diet": "Fish and aquatic invertebrates",
    "lifespan": "Up to 7 years",
    "description": "A large grebe with striking black and white plumage and a slender, long neck."
  },
  "054.Blue_Grosbeak": {
    "id": "054",
    "name": "Blue Grosbeak",
    "scientificName": "Passerina cyanea",
    "habitat": "Open woodlands and brushy fields",
    "diet": "Seeds, grains, and insects",
    "lifespan": "Up to 7 years",
    "description": "A striking blue bird with a large, conical bill and bold, dark markings."
  },
  "055.Evening_Grosbeak": {
    "id": "055",
    "name": "Evening Grosbeak",
    "scientificName": "Coccothraustes vespertinus",
    "habitat": "Woodlands and coniferous forests",
    "diet": "Seeds, fruits, and insects",
    "lifespan": "Up to 13 years",
    "description": "A large grosbeak with a yellow and black plumage, known for its powerful beak."
  },
  "056.Pine_Grosbeak": {
    "id": "056",
    "name": "Pine Grosbeak",
    "scientificName": "Pinicola enucleator",
    "habitat": "Coniferous forests and high altitudes",
    "diet": "Fruits, seeds, and buds",
    "lifespan": "Up to 10 years",
    "description": "A robust grosbeak with a large bill and colorful plumage, often found in colder regions."
  },
  "057.Rose_breasted_Grosbeak": {
    "id": "057",
    "name": "Rose-breasted Grosbeak",
    "scientificName": "Pheucticus ludovicianus",
    "habitat": "Woodlands, forests, and gardens",
    "diet": "Seeds, fruits, and insects",
    "lifespan": "Up to 9 years",
    "description": "A striking grosbeak with a distinctive rose-colored patch on its breast and a large, conical bill."
  },
  "058.Pigeon_Guillemot": {
    "id": "058",
    "name": "Pigeon Guillemot",
    "scientificName": "Cepphus columba",
    "habitat": "Coastal cliffs and rocky shores",
    "diet": "Fish and invertebrates",
    "lifespan": "Up to 15 years",
    "description": "A seabird with black plumage and a distinctive white wing patch, often found in rocky coastal habitats."
  },
  "059.California_Gull": {
    "id": "059",
    "name": "California Gull",
    "scientificName": "Larus californicus",
    "habitat": "Coastal areas, lakes, and inland wetlands",
    "diet": "Fish, insects, and small invertebrates",
    "lifespan": "Up to 30 years",
    "description": "A medium-sized gull with a light gray back and yellow legs, known for its adaptability to various habitats."
  },
  "060.Glaucous_winged_Gull": {
    "id": "060",
    "name": "Glaucous-winged Gull",
    "scientificName": "Larus glaucescens",
    "habitat": "Coastal areas, particularly along the Pacific coast",
    "diet": "Fish, mollusks, and scavenged food",
    "lifespan": "Up to 20 years",
    "description": "A large gull with a pale gray back and white underparts, often seen in coastal environments."
  },
  "061.Heermann_Gull": {
    "id": "061",
    "name": "Heermann Gull",
    "scientificName": "Larus heermanni",
    "habitat": "Coastal beaches and cliffs",
    "diet": "Fish, mollusks, and scavenged food",
    "lifespan": "Up to 25 years",
    "description": "A medium-sized gull with a distinctive dark red bill and pale gray plumage."
  },
  "062.Herring_Gull": {
    "id": "062",
    "name": "Herring Gull",
    "scientificName": "Larus argentatus",
    "habitat": "Coastal areas, lakes, and inland waters",
    "diet": "Fish, invertebrates, and carrion",
    "lifespan": "Up to 30 years",
    "description": "A large gull with pale gray wings and a yellow bill, often seen in urban and coastal environments."
  },
  "063.Ivory_Gull": {
    "id": "063",
    "name": "Ivory Gull",
    "scientificName": "Pagophila eburnea",
    "habitat": "Arctic regions, especially on sea ice",
    "diet": "Fish, invertebrates, and carrion",
    "lifespan": "Up to 15 years",
    "description": "A pure white gull found in polar regions, often associated with sea ice and glaciers."
  },
  "064.Ring_billed_Gull": {
    "id": "064",
    "name": "Ring-billed Gull",
    "scientificName": "Larus delawarensis",
    "habitat": "Coastal areas, lakes, and inland waterways",
    "diet": "Fish, insects, and garbage",
    "lifespan": "Up to 20 years",
    "description": "A medium-sized gull with a characteristic black ring around its bill and a versatile diet."
  },
  "065.Slaty_backed_Gull": {
    "id": "065",
    "name": "Slaty-backed Gull",
    "scientificName": "Larus schistisagus",
    "habitat": "Coastal and inland areas, especially in northern Asia",
    "diet": "Fish, invertebrates, and small mammals",
    "lifespan": "Up to 25 years",
    "description": "A large gull with a slaty gray back and a distinctive robust bill, primarily found in northern regions."
  },
  "066.Western_Gull": {
    "id": "066",
    "name": "Western Gull",
    "scientificName": "Larus occidentalis",
    "habitat": "Coastal beaches and cliffs along the Pacific coast",
    "diet": "Fish, mollusks, and carrion",
    "lifespan": "Up to 30 years",
    "description": "A large gull with a dark gray back and a robust bill, commonly found along the Pacific coastline."
  },
  "067.Anna_Hummingbird": {
    "id": "067",
    "name": "Anna Hummingbird",
    "scientificName": "Calypte anna",
    "habitat": "Gardens, shrubs, and woodlands in the western U.S.",
    "diet": "Nectar, small insects, and spiders",
    "lifespan": "Up to 9 years",
    "description": "A small hummingbird with vibrant magenta and green plumage, commonly found in western gardens."
  },
  "068.Ruby_throated_Hummingbird": {
    "id": "068",
    "name": "Ruby-throated Hummingbird",
    "scientificName": "Archilochus colubris",
    "habitat": "Woodlands, gardens, and forests in eastern North America",
    "diet": "Nectar, small insects, and spiders",
    "lifespan": "Up to 9 years",
    "description": "A small hummingbird with an iridescent ruby-red throat and green back, one of the most common hummingbirds in North America."
  },
  "069.Rufous_Hummingbird": {
    "id": "069",
    "name": "Rufous Hummingbird",
    "scientificName": "Selasphorus rufus",
    "habitat": "Gardens and forests in the western U.S. and Canada",
    "diet": "Nectar, small insects, and spiders",
    "lifespan": "Up to 5 years",
    "description": "A small hummingbird with a fiery red throat and greenish back, known for its aggressive territorial behavior."
  },
  "070.Green_Violetear": {
    "id": "070",
    "name": "Green Violetear",
    "scientificName": "Colibri thalassinus",
    "habitat": "Montane forests and gardens in Central America",
    "diet": "Nectar, small insects, and spiders",
    "lifespan": "Up to 10 years",
    "description": "A striking hummingbird with a vibrant green and violet plumage, commonly found in the highlands of Central America."
  },
  "071.Long_tailed_Jaeger": {
    "id": "071",
    "name": "Long-tailed Jaeger",
    "scientificName": "Stercorarius longicaudus",
    "habitat": "Arctic regions and coastal areas",
    "diet": "Fish, small birds, and eggs",
    "lifespan": "Up to 10 years",
    "description": "A large seabird with long, pointed tail feathers, found in the Arctic and known for its aggressive hunting behavior."
  },
  "072.Pomarine_Jaeger": {
    "id": "072",
    "name": "Pomarine Jaeger",
    "scientificName": "Stercorarius pomarinus",
    "habitat": "Coastal and open ocean regions",
    "diet": "Fish, small birds, and scavenged food",
    "lifespan": "Up to 15 years",
    "description": "A robust seabird with a distinctive pom-pom like tail, often seen harassing other birds for food."
  },
  "073.Blue_Jay": {
    "id": "073",
    "name": "Blue Jay",
    "scientificName": "Cyanocitta cristata",
    "habitat": "Woodlands, gardens, and parks in North America",
    "diet": "Nuts, seeds, fruits, and insects",
    "lifespan": "Up to 26 years",
    "description": "A striking bird with blue, white, and black plumage, known for its intelligence and complex vocalizations."
  },
  "074.Florida_Jay": {
    "id": "074",
    "name": "Florida Jay",
    "scientificName": "Cyanocitta cristata florincola",
    "habitat": "Florida pine forests and oak groves",
    "diet": "Acorns, seeds, fruits, and small insects",
    "lifespan": "Up to 15 years",
    "description": "A subspecies of the Blue Jay, with similar characteristics but found specifically in the Florida region."
  },
  "075.Green_Jay": {
    "id": "075",
    "name": "Green Jay",
    "scientificName": "Cyanocorax yncas",
    "habitat": "Woodlands and scrublands in southern Texas and Mexico",
    "diet": "Fruits, seeds, and insects",
    "lifespan": "Up to 15 years",
    "description": "A colorful jay with a vibrant green body and a striking blue head, found in the southwestern U.S. and Mexico."
  },
  "076.Dark_eyed_Junco": {
    "id": "076",
    "name": "Dark-eyed Junco",
    "scientificName": "Junco hyemalis",
    "habitat": "Forests, gardens, and mountainous areas across North America",
    "diet": "Seeds, insects, and small fruits",
    "lifespan": "Up to 11 years",
    "description": "A small sparrow with a dark-colored head and chest, commonly found in North American winter environments."
  },
  "077.Tropical_Kingbird": {
    "id": "077",
    "name": "Tropical Kingbird",
    "scientificName": "Tyrannus melancholicus",
    "habitat": "Tropical and subtropical regions of Central and South America",
    "diet": "Insects, small fruit, and nectar",
    "lifespan": "Up to 7 years",
    "description": "A large flycatcher with a bold yellow belly and gray wings, known for its aggressive defense of territory."
  },
  "078.Gray_Kingbird": {
    "id": "078",
    "name": "Gray Kingbird",
    "scientificName": "Tyrannus dominicensis",
    "habitat": "Tropical forests, mangroves, and coastal regions",
    "diet": "Insects, small lizards, and fruit",
    "lifespan": "Up to 5 years",
    "description": "A gray and white flycatcher with a distinctive black mask, found in tropical areas of the Caribbean."
  },
  "079.Belted_Kingfisher": {
    "id": "079",
    "name": "Belted Kingfisher",
    "scientificName": "Megaceryle alcyon",
    "habitat": "Rivers, lakes, and coastal areas in North America",
    "diet": "Fish, crustaceans, and aquatic insects",
    "lifespan": "Up to 10 years",
    "description": "A medium-sized kingfisher with a striking blue and white plumage, known for its skill in fishing."
  },
  "080.Green_Kingfisher": {
    "id": "080",
    "name": "Green Kingfisher",
    "scientificName": "Chloroceryle americana",
    "habitat": "Rivers, lakes, and wetlands in Central America",
    "diet": "Fish, insects, and crustaceans",
    "lifespan": "Up to 10 years",
    "description": "A small kingfisher with a vivid green back and a white belly, found near freshwater sources in Central America."
  },
  "081.Pied_Kingfisher": {
    "id": "081",
    "name": "Pied Kingfisher",
    "scientificName": "Ceryle rudis",
    "habitat": "Rivers, lakes, and wetlands across Africa and Asia",
    "diet": "Fish and small aquatic invertebrates",
    "lifespan": "Up to 10 years",
    "description": "A distinctive black-and-white kingfisher with a crest, known for its hovering flight while hunting."
  },
  "082.Ringed_Kingfisher": {
    "id": "082",
    "name": "Ringed Kingfisher",
    "scientificName": "Ceryle torquata",
    "habitat": "Rivers, lakes, and coastal regions in Central and South America",
    "diet": "Fish, crustaceans, and insects",
    "lifespan": "Up to 9 years",
    "description": "A large kingfisher with a distinctive white ring around its neck and a loud, harsh call."
  },
  "083.White_breasted_Kingfisher": {
    "id": "083",
    "name": "White-breasted Kingfisher",
    "scientificName": "Halcyon smyrnensis",
    "habitat": "Woodlands, gardens, and agricultural areas in Asia and parts of Africa",
    "diet": "Fish, small reptiles, and insects",
    "lifespan": "Up to 8 years",
    "description": "A brightly colored kingfisher with a blue back and white underparts, often found near water."
  },
  "084.Red_legged_Kittiwake": {
    "id": "084",
    "name": "Red-legged Kittiwake",
    "scientificName": "Rissa brevirostris",
    "habitat": "Coastal regions of Alaska and the Arctic",
    "diet": "Fish, crustaceans, and mollusks",
    "lifespan": "Up to 20 years",
    "description": "A seabird with red legs and a white body, often seen in the Arctic during the summer months."
  },
  "085.Horned_Lark": {
    "id": "085",
    "name": "Horned Lark",
    "scientificName": "Eremophila alpestris",
    "habitat": "Open fields, grasslands, and deserts across North America and Eurasia",
    "diet": "Seeds, insects, and small plants",
    "lifespan": "Up to 9 years",
    "description": "A small, ground-dwelling bird with distinctive black horns on its head, known for its melodious song."
  },
  "086.Pacific_Loon": {
    "id": "086",
    "name": "Pacific Loon",
    "scientificName": "Gavia pacifica",
    "habitat": "Coastal and inland waters in North America and East Asia",
    "diet": "Fish and aquatic invertebrates",
    "lifespan": "Up to 15 years",
    "description": "A medium-sized waterbird with a striking black and white plumage, known for its eerie calls."
  },
  "087.Mallard": {
    "id": "087",
    "name": "Mallard",
    "scientificName": "Anas platyrhynchos",
    "habitat": "Freshwater lakes, rivers, and marshes across North America, Europe, and Asia",
    "diet": "Aquatic plants, seeds, insects, and small fish",
    "lifespan": "Up to 10 years",
    "description": "A common duck species, with the male having an iridescent green head and a characteristic quack."
  },
  "088.Western_Meadowlark": {
    "id": "088",
    "name": "Western Meadowlark",
    "scientificName": "Sturnella neglecta",
    "habitat": "Grasslands and prairies in North America",
    "diet": "Insects, seeds, and berries",
    "lifespan": "Up to 12 years",
    "description": "A songbird known for its bright yellow chest and melodious song, often seen in open fields."
  },
  "089.Hooded_Merganser": {
    "id": "089",
    "name": "Hooded Merganser",
    "scientificName": "Lophodytes cucullatus",
    "habitat": "Freshwater lakes, rivers, and ponds across North America",
    "diet": "Fish, aquatic invertebrates, and small crustaceans",
    "lifespan": "Up to 10 years",
    "description": "A striking diving duck with a large, fan-shaped crest, known for its fish-catching abilities."
  },
  "090.Red_breasted_Merganser": {
    "id": "090",
    "name": "Red-breasted Merganser",
    "scientificName": "Mergus serrator",
    "habitat": "Coastal waters, rivers, and lakes across North America and Europe",
    "diet": "Fish and aquatic invertebrates",
    "lifespan": "Up to 12 years",
    "description": "A slender diving duck with a distinctive red breast and sharp, serrated bill, ideal for catching fish."
  },
  "091.Mockingbird": {
    "id": "091",
    "name": "Mockingbird",
    "scientificName": "Mimus polyglottos",
    "habitat": "Urban areas, forests, and suburban regions across North America",
    "diet": "Insects, berries, and small fruits",
    "lifespan": "Up to 8 years",
    "description": "A versatile bird known for its ability to mimic the sounds of other birds and even mechanical noises."
  },
  "092.Nighthawk": {
    "id": "092",
    "name": "Nighthawk",
    "scientificName": "Chordeiles minor",
    "habitat": "Open areas, deserts, and forests across North America",
    "diet": "Insects, particularly moths and beetles",
    "lifespan": "Up to 5 years",
    "description": "A nocturnal bird known for its distinctive flight pattern and haunting call during the evening hours."
  },
  "093.Clark_Nutcracker": {
    "id": "093",
    "name": "Clark's Nutcracker",
    "scientificName": "Nucifraga columbiana",
    "habitat": "Mountain forests in the western United States",
    "diet": "Pine seeds, insects, and berries",
    "lifespan": "Up to 15 years",
    "description": "A crow-like bird with a long, pointed bill, known for its ability to store pine seeds for later consumption."
  },
  "094.White_breasted_Nuthatch": {
    "id": "094",
    "name": "White-breasted Nuthatch",
    "scientificName": "Sitta carolinensis",
    "habitat": "Woodlands and forests across North America",
    "diet": "Insects, seeds, and nuts",
    "lifespan": "Up to 9 years",
    "description": "A small, acrobatic bird known for its habit of climbing trees headfirst while foraging for food."
  },
  "095.Baltimore_Oriole": {
    "id": "095",
    "name": "Baltimore Oriole",
    "scientificName": "Icterus galbula",
    "habitat": "Woodlands, forests, and suburban areas in eastern North America",
    "diet": "Nectar, fruits, and insects",
    "lifespan": "Up to 10 years",
    "description": "A vibrant orange and black bird known for its distinctive call and preference for hanging from branches while feeding."
  },
  "096.Hooded_Oriole": {
    "id": "096",
    "name": "Hooded Oriole",
    "scientificName": "Icterus cucullatus",
    "habitat": "Deserts, woodlands, and suburban areas in the southwestern United States",
    "diet": "Nectar, fruits, and insects",
    "lifespan": "Up to 12 years",
    "description": "A striking oriole with a bright yellow and black plumage, known for its large, hanging nests."
  },
  "097.Orchard_Oriole": {
    "id": "097",
    "name": "Orchard Oriole",
    "scientificName": "Icterus spurius",
    "habitat": "Woodlands, orchards, and farmlands across North America",
    "diet": "Fruits, insects, and nectar",
    "lifespan": "Up to 8 years",
    "description": "A smaller oriole species, with a dark red breast and a preference for fruit-laden trees."
  },
  "098.Scott_Oriole": {
    "id": "098",
    "name": "Scott's Oriole",
    "scientificName": "Icterus parisorum",
    "habitat": "Desert areas, particularly in the southwestern United States",
    "diet": "Nectar, fruits, and insects",
    "lifespan": "Up to 7 years",
    "description": "A yellow and black oriole, often found in arid environments, known for its loud, clear whistle."
  },
  "099.Ovenbird": {
    "id": "099",
    "name": "Ovenbird",
    "scientificName": "Seiurus aurocapilla",
    "habitat": "Woodlands and forests across North America",
    "diet": "Insects, spiders, and other small invertebrates",
    "lifespan": "Up to 6 years",
    "description": "A small, olive-colored bird known for its distinctive 'oven-shaped' nests built on the forest floor."
  },
  "100.Brown_Pelican": {
    "id": "100",
    "name": "Brown Pelican",
    "scientificName": "Pelecanus occidentalis",
    "habitat": "Coastal areas and estuaries across the Americas",
    "diet": "Fish and marine invertebrates",
    "lifespan": "Up to 16 years",
    "description": "A large, coastal bird with a distinctive long bill and a remarkable fishing technique that involves diving into the water."
  } 



}



# Model setup
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names)) 
state_dict = torch.load(r"new_bird_model.pth", map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return the prediction."""
    global model
    if model is None:
        model = torch.load('new_bird_model.pth', map_location=torch.device('cpu'))
    print("Request received at /predict") 
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            image = Image.open(filepath).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)

            predicted_class = class_names[predicted.item()]

            details = bird_details.get(predicted_class, None)

            response = {
                'class': predicted_class,
                'file_path': filepath,
                'details': details  # Include bird details in the response
            }

            return jsonify(response), 200

        except Exception as e:
            return jsonify({'error': f"An error occurred during prediction: {str(e)}"}), 500

    else:
        return jsonify({'error': 'Invalid file type. Allowed types are PNG, JPG, JPEG'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port)