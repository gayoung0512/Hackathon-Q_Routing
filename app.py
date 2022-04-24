from socket import *
import numpy as np
HOST = "127.0.0.1"
PORT = 9001
s = socket(AF_INET, SOCK_STREAM)
print ('Socket created')
s.bind((HOST, PORT))
print ('Socket bind complete')
s.listen(1)
print ('Socket now listening')
#print(s)

while True:
   #print("dong")
   #접속 승인
    conn, addr = s.accept()
    #print(conn)
    #print(addr)
    print("Connected by ", addr)
    # 글자수 세기
    idx = 0
   #데이터 수신
    rc = conn.recv(1024)
    rc = rc.decode("utf8").strip()
    if not rc: break
    for i in range(len(rc)):
        if rc[i]=="+":
            idx = i
            break
    rc1 = rc[0:idx]
    rc2 = rc[idx+1:]
        
    print("rc1:",rc1)
    print("rc2:",rc2)
    
    
    ori=rc1         #광개토관
    con=rc2
    print('ori:',ori)
    print('con:',con)

    print("추천 값 : 세종AI"  )

   #클라이언트에게 답을 보냄
    res = "AIcenter"
    conn.sendall(res.encode("utf-8"))
   #연결 닫기
    conn.close()
    break
s.close()
