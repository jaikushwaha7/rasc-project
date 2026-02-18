import json
import random

INPUT = "data/processed/vg_5k_subset.json"
OUTPUT = "data/processed/scene_graphs/t5_data.json"

def main():
    data = json.load(open(INPUT))
    samples = []

    for item in data:
        if "region_descriptions" not in item:
            continue

        captions = [r["phrase"] for r in item["region_descriptions"]]
        if not captions:
            continue

        obj_map = {o["object_id"]: o["names"][0] for o in item["objects"]}

        rels = []
        for r in item["relationships"]:
            s = obj_map.get(r["subject"]["object_id"])
            o = obj_map.get(r["object"]["object_id"])
            if s and o:
                rels.append(f"{s} {r['predicate']} {o}")

        if not rels:
            continue

        samples.append({
            "input": "; ".join(rels),
            "target": random.choice(captions)
        })

    json.dump(samples, open(OUTPUT, "w"), indent=2)
    print(f"Saved {len(samples)} caption pairs")

if __name__ == "__main__":
    main()


# import json
# import os

# INPUT = "data/processed/vg_5k_subset.json"
# OUTPUT = "data/processed/scene_graphs/t5_data.json"

# def main():
#     os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

#     data = json.load(open(INPUT))
#     samples = []

#     for item in data:
#         obj_map = {
#             o["object_id"]: o["names"][0].lower()
#             for o in item["objects"]
#         }

#         relations = []
#         for r in item["relationships"]:
#             s = obj_map.get(r["subject"]["object_id"])
#             o = obj_map.get(r["object"]["object_id"])
#             if s and o:
#                 relations.append(f"{s} {r['predicate']} {o}")

#         if not relations:
#             continue

#         graph_text = "; ".join(relations)

#         samples.append({
#             "input": graph_text,
#             "target": graph_text  # placeholder (can replace with human captions later)
#         })

#     json.dump(samples, open(OUTPUT, "w"), indent=2)
#     print(f"Saved {len(samples)} T5 samples")

# if __name__ == "__main__":
#     main()


