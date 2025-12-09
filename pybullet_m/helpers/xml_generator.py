import numpy as np
from xml.etree import ElementTree as ET


def get_fromto_coords(geom):
    return [float(coord) for coord in geom.get("fromto").split(" ")]


def get_size_coords(geom):
    return [float(el) for el in geom.get("size").split(" ")]


def get_pos_coords(body):
    body_pos = body.get("pos")
    if body_pos is None:
        return [0, 0, 0]
    else:
        return [float(coord) for coord in body_pos.split(" ")]


def make_coord_string(coords):
    return " ".join([str(c) for c in coords])


def get_length_coords(fromto_coords):
    lengths = [end - start for start, end in zip(fromto_coords[:3], fromto_coords[3:])]
    return lengths


def update_pos(body, pos_diff):
    pos_coords = get_pos_coords(body)
    new_pos_coords = [coord + diff for coord, diff in zip(pos_coords, pos_diff)]
    new_pos_str = make_coord_string(new_pos_coords)
    body.set("pos", new_pos_str)


def perturb_geom_size(geom, perturb, size_idx=0):
    base_geom_size_list = get_size_coords(geom)
    base_geom_size_list[size_idx] *= 1 + perturb
    new_geom_size = " ".join([str(el) for el in base_geom_size_list])
    geom.set("size", new_geom_size)


def perturb_cheetah_geom_size(geom, next_body, perturb):
    geom_length = get_size_coords(geom)[-1]
    perturb_geom_size(geom, perturb, size_idx=1)
    geom_pos = get_pos_coords(geom)
    geom_factor = np.linalg.norm(geom_pos)
    pos_diff = [geom_length * perturb * p / geom_factor for p in geom_pos]
    update_pos(geom, pos_diff)
    if next_body is not None:
        update_pos(next_body, [2 * p for p in pos_diff])


def fromto_z_shift(geom, shift):
    fromto_coords = get_fromto_coords(geom)
    fromto_coords[2] += shift
    fromto_coords[-1] += shift
    fromto_str = make_coord_string(fromto_coords)
    geom.set("fromto", fromto_str)


def perturb_geom_length(geom, perturb, center=None):
    fromto_coords = get_fromto_coords(geom)
    if center is None:
        length_coords = get_length_coords(fromto_coords)
        for i in range(3):
            fromto_coords[3 + i] += perturb * length_coords[i]
    else:
        center_center = center + center
        fromto_coords = [
            c * (1 + perturb) - perturb * cc
            for c, cc in zip(fromto_coords, center_center)
        ]

    new_fromto = make_coord_string(fromto_coords)
    geom.set("fromto", new_fromto)


def perturb_limb_length(upper_geom, lower_geom, lower_geom_root, ending_root, perturb):
    upper_geom_coords = get_fromto_coords(upper_geom)
    perturb_geom_length(upper_geom, perturb)
    new_upper_geom_coords = get_fromto_coords(upper_geom)
    upper_geom_coords_diff = [
        new - old for new, old in zip(new_upper_geom_coords, upper_geom_coords)
    ]
    upper_length_diff = get_length_coords(upper_geom_coords_diff)
    update_pos(lower_geom_root, upper_length_diff)
    lower_geom_coords = get_fromto_coords(lower_geom)
    perturb_geom_length(lower_geom, perturb)
    new_lower_geom_coords = get_fromto_coords(lower_geom)
    lower_geom_coords_diff = [
        new - old for new, old in zip(new_lower_geom_coords, lower_geom_coords)
    ]
    lower_length_diff = get_length_coords(lower_geom_coords_diff)
    if ending_root is not None:
        update_pos(ending_root, lower_length_diff)


def get_limb_length(upper_geom, lower_geom):
    upper_length = get_length_coords(get_fromto_coords(upper_geom))
    lower_length = get_length_coords(get_fromto_coords(lower_geom))
    limb_length = [upper + lower for upper, lower in zip(upper_length, lower_length)]
    return np.linalg.norm(limb_length)


def get_half_cheetah_leg_length(thigh, shin):
    thigh_length = get_size_coords(thigh)[-1]
    shin_length = get_size_coords(shin)[-1]
    return 2 * (thigh_length + shin_length)


def perturb_ant_xml(
    base_xml_path,
    out_path,
    torso_size_perturb=0,
    back_left_limb_size_perturb=0,
    back_left_limb_length_perturb=0,
    back_right_limb_size_perturb=0,
    back_right_limb_length_perturb=0,
    front_left_limb_size_perturb=0,
    front_left_limb_length_perturb=0,
    front_right_limb_size_perturb=0,
    front_right_limb_length_perturb=0,
):
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    torso_root = root.find("worldbody").find("body")
    torso = torso_root.find("geom")
    (
        front_right_hip_root,
        front_left_hip_root,
        back_left_hip_root,
        back_right_hip_root,
    ) = torso_root.findall("body")
    front_left_hip = front_left_hip_root.find("geom")
    front_left_leg_root = front_left_hip_root.find("body")
    front_left_leg = front_left_leg_root.find("geom")
    front_left_foot_root = front_left_leg_root.find("body")
    front_left_foot = front_left_foot_root.find("geom")
    front_right_hip = front_right_hip_root.find("geom")
    front_right_leg_root = front_right_hip_root.find("body")
    front_right_leg = front_right_leg_root.find("geom")
    front_right_foot_root = front_right_leg_root.find("body")
    front_right_foot = front_right_foot_root.find("geom")
    back_left_hip = back_left_hip_root.find("geom")
    back_left_leg_root = back_left_hip_root.find("body")
    back_left_leg = back_left_leg_root.find("geom")
    back_left_foot_root = back_left_leg_root.find("body")
    back_left_foot = back_left_foot_root.find("geom")
    back_right_hip = back_right_hip_root.find("geom")
    back_right_leg_root = back_right_hip_root.find("body")
    back_right_leg = back_right_leg_root.find("geom")
    back_right_foot_root = back_right_leg_root.find("body")
    back_right_foot = back_right_foot_root.find("geom")

    # Torso size
    perturb_geom_size(torso, torso_size_perturb)

    # Front left leg size
    perturb_geom_size(front_left_hip, front_left_limb_size_perturb)
    perturb_geom_size(front_left_leg, front_left_limb_size_perturb)
    perturb_geom_size(front_left_foot, front_left_limb_size_perturb)

    # Front left leg length
    perturb_limb_length(
        front_left_leg,
        front_left_foot,
        front_left_foot_root,
        None,
        front_left_limb_length_perturb,
    )

    # Front right leg size
    perturb_geom_size(front_right_hip, front_right_limb_size_perturb)
    perturb_geom_size(front_right_leg, front_right_limb_size_perturb)
    perturb_geom_size(front_right_foot, front_right_limb_size_perturb)

    # Front right leg length
    perturb_limb_length(
        front_right_leg,
        front_right_foot,
        front_right_foot_root,
        None,
        front_right_limb_length_perturb,
    )

    # Back left leg size
    perturb_geom_size(back_left_hip, back_left_limb_size_perturb)
    perturb_geom_size(back_left_leg, back_left_limb_size_perturb)
    perturb_geom_size(back_left_foot, back_left_limb_size_perturb)

    # Back left leg length
    perturb_limb_length(
        back_left_leg,
        back_left_foot,
        back_left_foot_root,
        None,
        back_left_limb_length_perturb,
    )

    # Back right leg size
    perturb_geom_size(back_right_hip, back_right_limb_size_perturb)
    perturb_geom_size(back_right_leg, back_right_limb_size_perturb)
    perturb_geom_size(back_right_foot, back_right_limb_size_perturb)

    # Back rigth leg length
    perturb_limb_length(
        back_right_leg,
        back_right_foot,
        back_right_foot_root,
        None,
        back_right_limb_length_perturb,
    )

    tree.write(out_path)


def perturb_half_cheetah_xml(
    base_xml_path,
    out_path,
    head_size_perturb=0,
    torso_size_perturb=0,
    torso_length_perturb=0,
    back_leg_size_perturb=0,
    back_leg_length_perturb=0,
    front_leg_size_perturb=0,
    front_leg_length_perturb=0,
):
    front_leg_size_perturb = max(-0.3, front_leg_size_perturb)
    back_leg_size_perturb = max(-0.3, back_leg_size_perturb)
    front_leg_length_perturb = max(-0.3, front_leg_length_perturb)
    back_leg_length_perturb = max(-0.3, back_leg_length_perturb)

    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    torso_root = root.find("worldbody").find("body")

    torso, head = torso_root.findall("geom")
    bthigh_root, fthigh_root = torso_root.findall("body")
    bthigh = bthigh_root.find("geom")
    bshin_root = bthigh_root.find("body")
    bshin = bshin_root.find("geom")
    bfoot_root = bshin_root.find("body")
    bfoot = bfoot_root.find("geom")
    fthigh = fthigh_root.find("geom")
    fshin_root = fthigh_root.find("body")
    fshin = fshin_root.find("geom")
    ffoot_root = fshin_root.find("body")
    ffoot = ffoot_root.find("geom")

    # Head size
    perturb_geom_size(head, head_size_perturb)

    # Torso size
    perturb_geom_size(torso, torso_size_perturb)

    # Torso length
    torso_length_coords = get_length_coords(get_fromto_coords(torso))
    perturb_geom_length(torso, torso_length_perturb)
    new_torso_length_coords = get_length_coords(get_fromto_coords(torso))
    pos_diff = [
        new - old for old, new in zip(torso_length_coords, new_torso_length_coords)
    ]
    update_pos(head, pos_diff)
    update_pos(fthigh_root, pos_diff)

    # Front leg size
    perturb_geom_size(fthigh, front_leg_size_perturb)
    perturb_geom_size(fshin, front_leg_size_perturb)
    perturb_geom_size(ffoot, front_leg_size_perturb)

    # Back leg size
    perturb_geom_size(bthigh, back_leg_size_perturb)
    perturb_geom_size(bshin, back_leg_size_perturb)
    perturb_geom_size(bfoot, back_leg_size_perturb)

    # Front leg length
    front_leg_length = get_half_cheetah_leg_length(fthigh, fshin)
    perturb_cheetah_geom_size(fthigh, fshin_root, front_leg_length_perturb)
    perturb_cheetah_geom_size(fshin, ffoot_root, front_leg_length_perturb)
    perturb_cheetah_geom_size(ffoot, None, front_leg_length_perturb)

    # Back leg length
    back_leg_length = get_half_cheetah_leg_length(bthigh, bshin)
    perturb_cheetah_geom_size(bthigh, bshin_root, back_leg_length_perturb)
    perturb_cheetah_geom_size(bshin, bfoot_root, back_leg_length_perturb)
    perturb_cheetah_geom_size(bfoot, None, back_leg_length_perturb)

    # Adjust the initial height
    height_diff = max(
        front_leg_length * front_leg_length_perturb,
        back_leg_length * back_leg_length_perturb,
    )
    update_pos(torso_root, [0, 0, 2 * height_diff])

    tree.write(out_path)


def perturb_walker_xml(
    base_xml_path,
    out_path,
    torso_size_perturb=0,
    right_leg_size_perturb=0,
    right_leg_length_perturb=0,
    left_leg_size_perturb=0,
    left_leg_length_perturb=0,
    right_foot_size_perturb=0,
    right_foot_length_perturb=0,
    left_foot_size_perturb=0,
    left_foot_length_perturb=0,
):
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    torso_root = root.find("worldbody").find("body")
    torso = torso_root.find("geom")
    right_thigh_root, left_thigh_root = torso_root.findall("body")
    right_thigh = right_thigh_root.find("geom")
    right_leg_root = right_thigh_root.find("body")
    right_leg = right_leg_root.find("geom")
    right_foot_root = right_leg_root.find("body")
    right_foot = right_foot_root.find("geom")
    left_thigh = left_thigh_root.find("geom")
    left_leg_root = left_thigh_root.find("body")
    left_leg = left_leg_root.find("geom")
    left_foot_root = left_leg_root.find("body")
    left_foot = left_foot_root.find("geom")

    # Torso size diff
    perturb_geom_size(torso, torso_size_perturb)

    # Right leg size
    perturb_geom_size(right_thigh, right_leg_size_perturb)
    perturb_geom_size(right_leg, right_leg_size_perturb)

    # Left leg size
    perturb_geom_size(left_thigh, left_leg_size_perturb)
    perturb_geom_size(left_leg, left_leg_size_perturb)

    # Right leg length
    right_leg_length = get_limb_length(right_thigh, right_leg)
    perturb_limb_length(
        right_thigh,
        right_leg,
        right_leg_root,
        right_foot_root,
        right_leg_length_perturb,
    )
    right_leg_length_diff = right_leg_length_perturb * right_leg_length

    # Left leg length
    left_leg_length = get_limb_length(left_thigh, left_leg)
    perturb_limb_length(
        left_thigh, left_leg, left_leg_root, left_foot_root, left_leg_length_perturb
    )
    left_leg_length_diff = left_leg_length_perturb * left_leg_length

    # Right foot size
    perturb_geom_size(right_foot, right_foot_size_perturb)

    # Left foot size
    perturb_geom_size(left_foot, left_foot_size_perturb)

    # Right foot length
    perturb_geom_length(right_foot, right_foot_length_perturb, center=[0.0, 0.0, 0.1])

    # Left foot length
    perturb_geom_length(left_foot, left_foot_length_perturb, center=[0.0, 0.0, 0.1])

    body_pos_diff = [0, 0, max(right_leg_length_diff, left_leg_length_diff)]
    update_pos(torso_root, body_pos_diff)

    tree.write(out_path)


def perturb_hopper_xml(
    base_xml_path,
    out_path,
    torso_size_perturb=0,
    leg_size_perturb=0,
    leg_length_perturb=0,
    foot_size_perturb=0,
    foot_length_perturb=0,
):
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    torso_root = root.find("worldbody").find("body")
    torso = torso_root.find("geom")
    thigh_root = torso_root.find("body")
    thigh = thigh_root.find("geom")
    leg_root = thigh_root.find("body")
    leg = leg_root.find("geom")
    foot_root = leg_root.find("body")
    foot = foot_root.find("geom")

    # Torso size diff
    perturb_geom_size(torso, torso_size_perturb)

    # Leg size
    perturb_geom_size(thigh, leg_size_perturb)
    perturb_geom_size(leg, leg_size_perturb)

    # Leg length
    leg_length = get_limb_length(thigh, leg)
    perturb_limb_length(thigh, leg, leg_root, foot_root, leg_length_perturb)
    leg_length_diff = leg_length_perturb * leg_length

    # Foot size
    perturb_geom_size(foot, foot_size_perturb)

    # Foot length
    perturb_geom_length(foot, foot_length_perturb, center=[0.0, 0.0, 0.1])

    body_pos_diff = [0, 0, leg_length_diff]
    update_pos(torso_root, body_pos_diff)

    tree.write(out_path)


def perturb_franka_urdf(base_path, out_path, length_scale=1.0, thickness_scale=1.0):
    """
    Franka Panda 로봇의 링크 길이와 두께를 변경하여 새로운 URDF로 저장합니다.
    - length_scale: 링크 길이 배율 (Joint Origin 이동 + Mesh Z축 스케일링)
    - thickness_scale: 링크 두께 배율 (Mesh X, Y축 스케일링)
    """
    tree = ET.parse(base_path)
    root = tree.getroot()

    # [핵심] 변형할 링크와 그 링크의 길이를 결정하는 '다음 관절'을 짝지어 정의합니다.
    # 구조: "Link 이름": "이 Link의 끝에 붙어있는 Joint 이름"
    # Panda는 주로 link2, link3, link4, link5가 팔의 길이를 담당합니다.
    perturbation_map = {
        "panda_link2": "panda_joint3", 
        "panda_link3": "panda_joint4",
        "panda_link4": "panda_joint5",
        "panda_link5": "panda_joint6"
    }

    # 1. 매핑된 파츠들을 순회하며 수정
    for link_name, next_joint_name in perturbation_map.items():
        
        # --- (1) 살(Mesh) 늘리기: 해당 Link의 메쉬 스케일 수정 ---
        for link in root.findall("link"):
            if link.get("name") == link_name:
                for tag in ["visual", "collision"]:
                    element = link.find(tag)
                    if element is None: continue
                    
                    geometry = element.find("geometry")
                    if geometry is None: continue
                    
                    mesh = geometry.find("mesh")
                    if mesh is not None:
                        # 기존 스케일 값 읽기 (없으면 1 1 1)
                        # Franka 메쉬는 Z축이 길이 방향인 경우가 많음
                        
                        # X, Y는 두께(thickness), Z는 길이(length)를 적용
                        # 시각적으로 틈이 안 생기게 하려면 Z축도 늘려야 함!
                        new_scale = f"{thickness_scale} {thickness_scale} {length_scale}"
                        mesh.set("scale", new_scale)

        # --- (2) 뼈(Joint) 늘리기: 다음 관절(Next Joint)의 위치 밀어내기 ---
        for joint in root.findall("joint"):
            if joint.get("name") == next_joint_name:
                origin = joint.find("origin")
                if origin is None: continue
                
                # 현재 관절의 위치(xyz) 벡터 가져오기
                xyz = [float(x) for x in origin.get("xyz").split()]
                
                # 벡터 전체에 스케일을 곱해 거리를 멀어지게 함
                # (Z축뿐만 아니라 꺾여있는 관절의 경우 벡터 방향대로 밀어내야 함)
                new_xyz = [x * length_scale for x in xyz]
                
                origin.set("xyz", f"{new_xyz[0]} {new_xyz[1]} {new_xyz[2]}")

    tree.write(out_path)