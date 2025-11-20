def is_valid():
    T = int(input())
    results = []
    for _ in range(T):
        n, m = map(int, input().split())
        pair1 = list(map(int, input().split()))
        pair2 = list(map(int, input().split()))
        diff_pairs = set()
        valid = True
        for i in range(m):
            x = pair1[i]
            y = pair2[i]
            if x == y:  # 攻击不能与自身差别
                valid = False
                break
            diff_pairs.add((x, y))
            if (y, x) not in diff_pairs:  # 检查对称性
                valid = False
                break
        results.append("Yes" if valid else "No")
    for res in results:
        print(res)

is_valid()