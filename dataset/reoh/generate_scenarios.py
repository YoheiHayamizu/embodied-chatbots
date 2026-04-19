#!/usr/bin/env python3
# generate_scenarios_jsonl.py

import os
import json
import argparse
import random

# 物件からランダムに選ぶことのできる特徴リスト
FEATURES = [
    ("entrance", "storage"),
    ("entrance", "size"),
    ("livingroom", "flooring"),
    ("livingroom", "size"),
    ("livingroom", "view"),
    ("bedroom", "bedtype"),
    ("bedroom", "size"),
    ("bedroom", "view"),
    ("kitchen", "kitchentype"),
    ("kitchen", "size"),
    ("kitchen", "view"),
]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def build_agent_description(inform: dict) -> str:
    """
    Build the agent.description HTML from inform mapping.
    Special-case 'storage' to use natural phrasing.
    """
    desc = (
        "You are a real estate agent. You have to provide a potential buyer with the following "
        "information and answer any questions. Try to build a good relationship with the buyer "
        "and make them interested in the property.<br>"
        "The features to inform are: <br><ul>\n"
    )
    for room, feats in inform.items():
        for feat, val in feats.items():
            if feat == "storage":
                # natural phrasing for storage yes/no
                if str(val).lower() == "yes":
                    desc += f"  <li><span class='emphasis'> the {room} has storage </span></li>\n"
                else:
                    desc += f"  <li><span class='emphasis'> the {room} has no storage </span></li>\n"
            else:
                desc += f"  <li><span class='emphasis'> the {feat} of the {room} is {val} </span></li>\n"
    desc += "</ul>."
    return desc


def build_buyer_description(persona: dict, request: dict) -> str:
    """
    Build the buyer.description HTML from persona and request mapping.
    """
    desc = "You are a potential buyer and are interested in buying a property.<br>Your persona:<ul>\n"
    # list out all persona sentences
    for sentence in persona.values():
        desc += f"  <li>{sentence}</li>\n"
    desc += "</ul><br><br>You have to ask the agent for the following information:<br><ul>\n"
    # list each requested feature
    for room, feats in request.items():
        for feat in feats.keys():
            desc += f"  <li><span class='emphasis'> the {feat} of the {room} </span></li>\n"
    desc += "</ul><br>Also inform the buyer's personal information to the agent to build a good relationship."
    return desc


def main(properties_path, personas_path, output_path):
    # Load input files
    props_data = load_json(properties_path)
    personas = load_json(personas_path)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    counter = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for prop in props_data["properties"]:
            rooms = prop["rooms"]
            # この物件に利用可能な特徴だけをフィルタ
            available = [(r, f) for (r, f) in FEATURES if f in rooms[r]]
            for persona in personas:
                # エージェントのゴールをランダムに2〜3個選択
                chosen = random.sample(available, k=random.choice([2, 3]))
                inform = {}
                request = {}
                for room, feat in chosen:
                    val = rooms[room][feat]
                    inform.setdefault(room, {})[feat] = val
                    # request.setdefault(room, {})[feat] = ""  # requestは空で良い

                agent_goal = {"description": build_agent_description(inform), "inform": inform, "request": request}

                # userのゴールはpersonaのgoal.informとrequestをそのまま使う
                user_goal = {
                    "description": build_buyer_description(persona["persona"], request),
                    "inform": persona["goal"]["inform"],
                    "request": persona["goal"]["request"],
                }

                scenario = {
                    "scenario_id": f"scenario-{counter:04d}",
                    "property": prop,
                    "persona": persona,
                    "goals": {"agent": agent_goal, "user": user_goal},
                }

                # Agent の inform と Buyer の request から回るべき部屋を順序付けて domains リストに
                domains = []
                # まず Buyer のゴールの部屋
                for domain in list(user_goal["request"].keys()) + list(user_goal["inform"].keys()):
                    if domain not in domains:
                        domains.append(domain)
                # 次に Agent のゴールの部屋
                for domain in list(agent_goal["inform"].keys()) + list(agent_goal["request"].keys()):
                    if domain not in domains:
                        domains.append(domain)
                # 先頭に常に entrance を追加
                domains = ["entrance"] + domains
                # scenario に domains をセット
                scenario["domains"] = list(set(domains))

                out_f.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                print(f"[+] Wrote {scenario['scenario_id']}")
                counter += 1

    print(f"\nFinished: {counter} scenarios written to {output_path}")

    # JSONLファイルを読み込んで、一行をJSONファイルに変換して別で保存
    convert_jsonl_to_json(output_path)
    print(f"[+] Converted {output_path} to JSON files.")


def convert_jsonl_to_json(output_path):
    """
    JSONLファイルを読み込んで、一行をJSONファイルに変換して別で保存
    Args:
        output_path (str): JSONLファイルのパス
    """
    # JSONLファイルが存在しない場合は終了
    if not os.path.exists(output_path):
        print(f"[-] JSONL file {output_path} does not exist.")
        return
    if not os.path.isfile(output_path):
        print(f"[-] {output_path} is not a valid file.")
        return

    # JSONLファイルを読み込んで、一行をJSONファイルに変換して別で保存
    json_output_path_all = os.path.join(os.path.dirname(output_path))
    json_output_path_one = os.path.join(os.path.dirname(output_path), "scenarios")
    if not os.path.exists(json_output_path_one):
        os.makedirs(json_output_path_one, exist_ok=True)

    scenarios = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            scenario = json.loads(line)
            scenario_id = scenario["scenario_id"]
            property_id = scenario["property"]["id"]
            persona_id = scenario["persona"]["id"]
            scenarios.append(scenario)

    for scenario in scenarios:
        # JSONファイルに保存\
        scenario_id = scenario["scenario_id"]
        property_id = scenario["property"]["id"]
        persona_id = scenario["persona"]["id"]
        json_filename = os.path.join(json_output_path_one, f"{scenario_id}-{property_id}-{persona_id}.json")
        save_json(scenario, json_filename)
        print(f"[+] Wrote {json_filename}")

    # For all scenarios
    json_output_path_all = os.path.join(json_output_path_all, "all-scenarios.json")
    save_json(scenarios, json_output_path_all)
    print(f"[+] Wrote {json_output_path_all}")

    print(f"\nFinished: {len(scenarios)} scenarios written to {json_output_path_all}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a JSONL file of scenarios from properties.json + personas.json")
    parser.add_argument("-p", "--properties", default="assets/properties.json", help="Path to properties.json")
    parser.add_argument(
        "-u",
        "--personas",
        default="assets/personas.json",
        help="Path to personas.json (with embedded goal.inform & goal.request)",
    )
    parser.add_argument("-o", "--output", default="scenarios.jsonl", help="Output path for scenarios.jsonl")
    args = parser.parse_args()
    main(args.properties, args.personas, args.output)
