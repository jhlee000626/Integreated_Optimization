import json

with open(r'c:\Users\user\Desktop\Integrated_Optimization\output\verification\case3\case_artifact.json') as f:
    d = json.load(f)

sch = d['milp_result']['schedule']
print("t | P_req  | DG1 DG2 DG3 DG4 | DG1_P  DG2_P  DG3_P  DG4_P  | P_dc  P_c   | SOC")
for s in sch:
    t = s['t']
    pr = s['P_req_MW']
    d1 = s['DG1_ON']
    d2 = s['DG2_ON']
    d3 = s['DG3_ON']
    d4 = s['DG4_ON']
    p1 = s['DG1_P_MW']
    p2 = s['DG2_P_MW']
    p3 = s['DG3_P_MW']
    p4 = s['DG4_P_MW']
    pdc = s['P_dc_MW']
    pc = s['P_c_MW']
    soc = s['SOC']
    print(f"{t:2d} | {pr:6.2f} | {d1}   {d2}   {d3}   {d4}  | {p1:6.2f} {p2:6.2f} {p3:6.2f} {p4:6.2f} | {pdc:5.2f} {pc:5.2f} | {soc:.3f}")
