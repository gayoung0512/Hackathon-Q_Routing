import math
import cv2
import csv
from collections import Counter
import copy
import numpy as np
import random
import pandas as pd
import time

#--------------------Q-routing-------------------
def get_dict(data):
    # 값 리스트로 변환해 저장
    A_0 = data["original"].values.tolist()
    Z_0 = data["connected"].values.tolist()
    weight = data["weight"].values.tolist()

    # csv: 중복이 제거된 상태 => 쌍방
    A = A_0 + Z_0  # original+connected
    Z = Z_0 + A_0  # connected+original
    weight = weight + weight
    A = list(map(lambda x: int(x), A))  # A를 int형으로 바꿔서 리스트로 저장
    Z = list(map(lambda x: int(x), Z))  # Z를 int형으로 바꿔서 리스트로 저장

    A_key = sorted(set(A))  # A를 오름차순 목록으로 정렬(중복 제거) A: 전체 노드
    A_Z_dict = {}  # 딕셔너리 선언

    for i in range(len(A_key)):
        A_Z_dict[A_key[i]] = []  # key 개수에 맞게 배열 생성

    for i in range(len(A)):
        if Z[i] not in A_Z_dict[A[i]]:  # 중복 제외하고 나머지 저장
            A_Z_dict[A[i]].append(Z[i])

    return {"A": A,
            "Z": Z,
            "weight": weight,
            "A_Z_dict": A_Z_dict}
#get_R_Q


def initial_R(A,Z,weight,A_Z_dict):

    R = {} #R 딕셔너리 선언
    net = copy.deepcopy(A_Z_dict) #net에 A_Z_dict 사본 반환
    for i in net.keys():
        sub_key = net[i] # sub_key에 A_Z_dict[key] 저장
        sub_dic = {} #딕셔너리 선언
        for j in sub_key:#sub_key : 연결 가능한 노드 저장
            sub_dic[j] = 0 #sub_dict[key]에다가 0
        R[i] = sub_dic# R[A_Z_dict.key]=sub_dict 배열(연결 가능한 노드에 0이 들어간 배열) 추가
    for i in range(len(A)):
        R[A[i]][Z[i]] = weight[i]#R배열에 가중치 저장 ex) R[0][3]=1, 노드 0와 3 사이 거리 :1
    return R

def initial_Q(R):

    Q = copy.deepcopy(R)#Q에 R 배열 사본 저장
    for i in Q.keys(): #노드
        for j in Q[i].keys(): #각 노드와 연결 가능한 노드
            Q[i][j] = 100 #가중치에 100 저장
    return Q


# get_all_routes

# single_dict : 가장 작은 가중치를 가진 노드 정보 저장하는 함수
def get_single_dict(dic):  # dic : Q[i] 받아옴

    single_link = {}  # single_link 딕셔너리 선언
    min_value = min(dic.values())  # Q[i].values 즉 저장된 가중치들 중 가장 작은 수 저장
    for key in dic.keys():
        if dic[key] == min_value:
            single_link[key] = dic[key]  # 가장 작은 가중치(value)와 key 모두 저장
    return single_link.keys()  # 노드의 key를 반환


# get_best_nodes : start-> end 경로 반환 함수
# update Q[i] 가중치가 가장 작은 Q[i]부터 저장된 리스트
def get_best_nodes(Q, start, end):

    next_level = [start]  # 시작점 저장-> 배열로 변환
    node_use = [start]  # 시작점 저장
    while list(set(next_level) & set(end)) == []:  # next_level&end 교집합 아무것도 없는 경우
        temp_level = []
        for i in next_level:
            temp_level += get_single_dict(Q[i])  # temp_level에 가장 작은 가중치가 저장된 Q[i][j] 추가
        next_level = list(set(temp_level))  # next_level을 가장 가중치가 작은 Q[i][j]로 저장
        node_use += next_level  # node_use에도 next level 추가: 거쳐간 노드 정보 저장
    return list(set(node_use))  # for문 거쳐간 node_use 리스트(거쳐간 노드 정보) 반환 -> 경로(start->end) 반환


# get_best_net: 경로 노드들 = 가중치가 가장 작은 인접한 노드들만 저장
def get_best_net(Q, nodes):  # nodes: 경로 정보 저장

    best_net = {}
    for i in nodes:  # ex) nodes: 0->3->5->7
        best_net[i] = list(set(get_single_dict(Q[i])) & set(nodes))  # 가중치 가장 작은 노드 정보 & nodes의 교집합 출력
        # 경로 정보 내에서 가중치가 가장 작은 인접한 노드들을 저장
    return best_net


def get_all_best_routes(graph, start, end, max_depth):

    # max_depth=route_len+1(최단 거리로 선택한 경로의 길이 +1)
    past_path = []
    queue = []
    queue.append([start])  # queue에 시작점을 넣어둠
    while queue:  # 큐가 비어있지 않다면 반복 실행
        path = queue.pop(0)  # path에 queue의 첫번째 원소 저장하고 queue에서는 삭제
        node = path[-1]  # path의 마지막 노드 저장

        for adjacent in graph.get(node, []):  # get([,])받아오고 싶은 키 값, 값이 없을 때 가져올 디폴트 값
            new_path = list(path)
            if adjacent in end:  # 인접 노드가 end인 경우
                new_path.append(adjacent)
                past_path.append(new_path)
                continue

            if adjacent in new_path:  # 중복되는 경우 pass
                continue

            new_path.append(adjacent)  # new pass에 인접노드 추가
            if len(new_path) >= max_depth and new_path[-1] not in end:  # (인접한 노드 中 최단 거리 선택한 경로의 길이+1) 이상이 되었는데 end에 도달하지 못 한 경우
                break

            queue.append(new_path)  # queue에 path 저장
            past_path.append(new_path)  # 경로 정보 저장

    best_paths = []
    for l in past_path:
        if l[-1] in end:  # 마지막에 지정된 도착점에 도착하는 경로만 best_path에 저장
            best_paths.append(l)
    return best_paths


def get_cost(R, route):
    cost = 0
    for i in range(len(route) - 1):
        cost += R[route[i]][route[i + 1]]
    return round(cost, 3)


# routes: best routes 받아온 거
def count_routes(routes):
    ends_find = []
    all_routes = {}
    for i in range(len(routes)):  # routes 안에 있는 배열 개수만큼
        ends_find.append(routes[i][-1])  # end_find에 best routes의 마지막 요소 저장

    count = dict(Counter(ends_find))  # ends_find에서 각 개체가 몇 번 나오는지 count

    ends = list(set(ends_find))  # 중복 요소 제거 후 리스트로 구성

    for i in ends:
        all_routes[i] = []  # all_routes 2차원 배열 구성
    for i in routes:
        end = i[-1]  # routes의 마지막 요소(end) 저장
        all_routes[end].append(i)  # all_routes[각 path의 end node]=end node에 도착하는 path들을 저장
    return {"routes_number": count,  # 최적의 경로 개수
            "all_routes": all_routes}  # 도착점에 도착하는 최적의 경로들 저장


# get_route: start부터 최단 거리 선택하여 경로 탐색
def get_route(Q, start, end):

    """ input is  Q-table is like:{1: {2: 0.5, 3: 3.8},
                                   2: {1: 5.9, 5: 10}} """
    single_route = [start]
    while single_route[-1] not in end:  # 마지막 노드가 end와 다르다면
        next_step = min(Q[single_route[-1]], key=Q[single_route[-1]].get)  # single_route[-1]와 연결된 가중치가 가장 작은 노드를 반환
        single_route.append(next_step)  # single route에 next_step 저장 => 최단 경로 저장
        if len(single_route) > 2 and single_route[-1] in single_route[:-1]:  # 노드 경로가 중복되는 예외 경우 제외
            break
    return single_route


# Q_routing


def update_Q(T, Q, current_state, next_state, alpha):
    current_t = T[current_state][next_state]  # ex) 0에서 2면 [0][2] 가중치
    current_q = Q[current_state][next_state]  # 현재 노드와 다음 노드 사이의 가중치 -> 이전의 측정치들
    # Q-learning 학습 방향을 결정하는 gradient 추정식
    new_q = current_q + alpha * (current_t + min(Q[
                                                     next_state].values()) - current_q)  # current_q+learning rate(거리 가중치+ 다음 노드의 연결 가능 노드 중 가중치가 가장 작은 것)-current_q)
    # 예측+learning rate*(실제값 - 예측값) 을 통해 최솟값 스스로 학습시키기
    # 예측값이 크면 작아지게, 예측값이 작으면 커지게

    Q[current_state][next_state] = new_q  # Q에 new_q 저장 : 새로운 가중치 저장
    return Q


def get_min_state(dic, valid_moves):
    """input dic is like {3: -0.5, 10: -0.1}
    valid_moves is like [1,3,5]"""
    new_dict = dict((k, dic[k]) for k in valid_moves)
    return min(new_dict, key=new_dict.get)


def get_route(Q, start, end):
    """ input is  Q-table is like:{1: {2: 0.5, 3: 3.8},
                                   2: {1: 5.9, 5: 10}} """

    single_route = [start]  # start
    while single_route[-1] not in end:
        next_step = min(Q[single_route[-1]], key=Q[single_route[-1]].get)
        single_route.append(next_step)
        if len(single_route) > 2 and single_route[-1] in single_route[:-1]:
            break
    return single_route


def get_key_of_min_value(dic):
    min_val = min(dic.values())
    return [k for k, v in dic.items() if v == min_val]


# 받아온 인자: R,Q,alpha,epsilon,n_episodes,start,end

# alpha = 0.7 # learning rate
# epsilon = 0.1 #greedy policy
# n_episodes = 1000

def Q_routing(T, Q, alpha, epsilon, n_episodes, start, end):

    nodes_number = [0, 0]

    # 학습시키기 위해
    for e in range(n_episodes):  # 0부터 1000까지

        current_state = start  # 시작점 저장

        goal = False
        while not goal:  # While True
            valid_moves = list(Q[current_state].keys())  # 시작점 노드와 연결 가능한 노드를 valid_moves 리스트에 저장

            if len(valid_moves) <= 1:  # 연결 가능한 노드의 개수가 1개 이하면
                next_state = valid_moves[
                    0]  # valid_moves[0]: 다음 노드 ex)0->1까지 연결가능한 노드: 2 => 0->2->1 / 연결 가능한 노드: 없음 => 0->1
            else:
                best_action = random.choice(get_key_of_min_value(Q[current_state]))  # 가중치가 가장 작은 것들 중 아무거나 저장
                # 그리디 정책: 랜덤 난수가 0.1보다 작을 때 무작위 행동 선택
                if random.random() < epsilon:  # random.random : 0부터 1사이 random 실수 return
                    valid_moves.pop(
                        valid_moves.index(best_action))  # 가중치 가장 작은 노드가 연결 가능한 노드들 중 몇번째에 위치한지 (index 받아서 삭제)
                    next_state = random.choice(valid_moves)  # 연결 가능한 노드들 중 아무거나 next_state 선택
                # 큰 경우 최적의 행동 선택
                else:
                    next_state = best_action  # best_action을 next_node로 설정

            Q = update_Q(T, Q, current_state, next_state, alpha)  # Q(current_state- next_state) update

            if next_state in end:  # 노드가 도착점과 일치할 때
                goal = True
            current_state = next_state  # 현재 노드를 이동할 노드로 바꿔옴

        if e in range(0, 1000, 200):  # (0,200,400,600,800)
            for i in Q.keys():
                for j in Q[i].keys():
                    Q[i][j] = round(Q[i][j], 6)  # 소수점 여섯 자리까지
            nodes = get_best_nodes(Q, start, end)  # update된 Q 바탕으로 start부터 end 경로 정보 nodes에 저장
            nodes_number.append(len(nodes))  # nodes_number = [0,0] 에 경로의 노드 개수 추가  ex)[0,0,3]->[0,0,3,5]->
            #print("nodes:", nodes_number)
            # 더이상 최적의 경로가 산출되지 않을 경우
            if len(set(nodes_number[-3:])) == 1:  # 마지막 세 개 원소에서 중복 제외하고 하나 남으면 break
                break
    return Q

def get_result(R,Q,alpha,epsilon,n_episodes,start,end):

    Q = Q_routing(R,Q,alpha,epsilon,n_episodes,start,end)
    #Q 완전히 update 된 상태 -> 학습완료된 상태
    nodes = get_best_nodes(Q,start,end)##경로 정보 nodes에 저장
    graph = get_best_net(Q,nodes) #최적 인접 노드를 이용한 경로 정보 저장
    route_len = len(get_route(Q,start,end)) #최단 거리로 선택한 경로의 길이 저장
    routes = get_all_best_routes(graph,start,end,route_len+1) #routes에 best routes 저장
    result = count_routes(routes) #최적의 경로와 최적의 경로 개수 저장
    ###
    ends_find = []
    for i in range(len(routes)):
        ends_find.append(routes[i][-1])
    ends_find = list(set(ends_find))
    ###
    cost = []
    for i in routes:
        cost.append(get_cost(R,i))
    Counter(cost)
    return {"nodes":nodes,
            "graph":graph,
            "ends_find":ends_find,
            "cost":dict(Counter(cost)),
            "routes_number":result['routes_number'],
            "all_routes":result['all_routes']}
#main.py
data = pd.read_csv("graph.csv")
graph = get_dict(data)


A = graph["A"] #original+connected
Z = graph["Z"] #connected+original
weight = graph["weight"] #weight+weight
A_Z_dict = graph["A_Z_dict"] #노드별로 연결 가능한 노드
#print(A)
#print(A_Z_dict)

##
start = 0
end = [65]

#get_R_Q

R = initial_R(A,Z,weight,A_Z_dict) #2차원 딕셔너리 : R에 각 노드별로 연결 가능한 노드들과의 거리 가중치 저장
Q = initial_Q(R) # R과 같음 : 대신 거리 가중치 대신 100 저장

alpha = 0.7 # learning rate
epsilon = 0.1 #greedy policy
n_episodes = 1000

time0 = time.time() #컴퓨터의 현재 시각 구하는 함수
result = get_result(R,Q,alpha,epsilon,n_episodes,start,end)

print("최적 경로: ", result["all_routes"])
