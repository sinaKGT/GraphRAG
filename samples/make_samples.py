"""Generate four FICTIONAL, related sample datasets for the Datasets page.
All names, IDs, places and numbers are synthetic. Shared keys:
  junctions.junction_id  <- traffic_readings.junction_id, schools.nearest_junction_id
  junctions.ward         <- schools.ward, emergency_staff.coverage_ward
Usage:  python make_samples.py [out_dir]   (needs openpyxl)"""
import random
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

from openpyxl import Workbook

rnd = random.Random(7)
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "datasets")
out.mkdir(parents=True, exist_ok=True)

WARDS = ["Riverside", "Northgate", "Hillcrest", "Meadowbank", "Castlefield", "Eastbrook", "Westfield", "Southmoor"]
STREETS = ["Elm Road", "Station Street", "Mill Lane", "Park Avenue", "Canal Way", "Church Street", "Victoria Road",
           "Market Street", "Bridge Road", "Orchard Lane", "High Street", "Queens Drive", "Kingsway", "Abbey Road",
           "Forest Road", "Meadow Lane", "London Road", "Derby Road", "Hill Street", "Riverside Walk"]


def save(name, header, rows):
    wb = Workbook(write_only=True)
    ws = wb.create_sheet(name.replace(".xlsx", "")[:31])
    ws.append(header)
    for r in rows:
        ws.append(r)
    wb.save(out / name)
    print("wrote", out / name)


# 1) junctions --------------------------------------------------------------------------
junctions = []
for i in range(1, 151):
    a, b = rnd.sample(STREETS, 2)
    junctions.append([f"J{i:04d}", f"{a} / {b}", round(52.93 + rnd.random() * 0.06, 6), round(-1.20 + rnd.random() * 0.10, 6),
                      rnd.choice(WARDS), rnd.choices(["A road", "B road", "Local"], [3, 3, 4])[0],
                      f"SC-{rnd.randint(1000, 9999)}", rnd.choice([20, 30, 30, 40]), rnd.randint(1, 4),
                      rnd.choice(["Signalised", "Roundabout", "Priority"])])
save("junctions.xlsx", ["junction_id", "junction_name", "lat", "lon", "ward", "road_class", "signal_controller_id",
                        "speed_limit_mph", "lanes", "control_type"], junctions)

# 2) traffic readings (large) -------------------------------------------------------------
def readings():
    t0 = datetime(2026, 9, 1)
    rid = 0
    for step in range(800):                       # 800 x 5 min ~ 2.8 days
        ts = t0 + timedelta(minutes=5 * step)
        peak = 1.8 if ts.hour in (8, 9, 16, 17) else (0.3 if ts.hour < 6 else 1.0)
        for j in junctions:
            rid += 1
            veh = max(0, int(rnd.gauss(60, 15) * peak))
            yield [rid, j[0], ts, veh, int(veh * rnd.uniform(0.02, 0.12)), round(min(64, max(3, rnd.gauss(38 / peak, 6))), 1),
                   round(min(100, veh / 1.5 + rnd.uniform(-5, 5)), 1), rnd.choices(["OK", "DEGRADED", "OFFLINE"], [96, 3, 1])[0]]
save("traffic_readings.xlsx", ["reading_id", "junction_id", "timestamp", "vehicle_count", "hgv_count", "avg_speed_kmh",
                               "occupancy_pct", "sensor_status"], readings())

# 3) schools --------------------------------------------------------------------------------
PRE = ["Oakfield", "Willow Bank", "Ashgrove", "Brookside", "Larchwood", "Fernhill", "Kingfisher", "Maple Tree", "Heron Way",
       "Beechcroft", "Holly Park", "Rosewood", "Silverdale", "Thornbury", "Cedar Lodge"]
schools = []
for i in range(1, 61):
    j = rnd.choice(junctions)
    phase = rnd.choice(["Primary", "Primary", "Secondary"])
    schools.append([f"S{i:03d}", f"{rnd.choice(PRE)} {phase} School", phase,
                    rnd.randint(180, 420) if phase == "Primary" else rnd.randint(700, 1600),
                    round(j[2] + rnd.uniform(-0.003, 0.003), 6), round(j[3] + rnd.uniform(-0.003, 0.003), 6), j[0], j[4],
                    time(8, rnd.choice([30, 40, 45])), time(15, rnd.choice([0, 15, 20, 30])),
                    f"{rnd.choice(STREETS)} playing field", rnd.choice(["Yes", "No"])])
save("schools.xlsx", ["school_id", "school_name", "phase", "pupils", "lat", "lon", "nearest_junction_id", "ward",
                      "start_time", "finish_time", "evacuation_point", "has_send_unit"], schools)

# 4) emergency / responsibility staff ---------------------------------------------------------
FIRST = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Jamie", "Robin", "Avery", "Riley", "Quinn", "Drew",
         "Rowan", "Harper", "Ellis", "Reese", "Skyler", "Charlie", "Frankie", "Logan"]
LAST = ["Ashdown", "Brightwell", "Carrow", "Dunmore", "Elsworth", "Fairley", "Gorrell", "Hatherley", "Ingram", "Jessop",
        "Kirkby", "Lindell", "Marlow", "Norbury", "Oakes", "Pendle", "Quarry", "Radley", "Stanton", "Thorne"]
ROLES = [  # organisation, role, responsibility
    ("Fire & Rescue Service", "Fire Safety Officer", "Fire incidents, building evacuation and hazardous material response"),
    ("Fire & Rescue Service", "Station Commander", "Command of fire crews and incident scene control"),
    ("City Council Highways", "Traffic Control Officer", "Road closures, diversions and signal timing changes"),
    ("City Council Highways", "Highways Inspector", "Road surface damage, structural street furniture and bridges"),
    ("City Council Highways", "Network Duty Manager", "Coordinating the road network during major incidents"),
    ("Police", "Roads Policing Officer", "Collision scene management and traffic enforcement"),
    ("Ambulance Service", "Paramedic Team Lead", "Casualty treatment and hospital transfer"),
    ("Gas Network Operator", "Gas Emergency Engineer", "Gas leaks and isolation of gas mains"),
    ("Water Utility", "Water Network Technician", "Burst mains, flooding and water supply isolation"),
    ("Electricity Network Operator", "Grid Field Engineer", "Substations, underground cables and power outages"),
    ("City Council Education", "School Liaison Officer", "Contacting schools, pupil safety and school closures"),
    ("City Council Emergency Planning", "Emergency Planning Officer", "Multi-agency coordination and public warnings"),
]
staff = []
for i in range(1, 421):
    org, role, resp = rnd.choice(ROLES)
    staff.append([f"E{i:04d}", f"{rnd.choice(FIRST)} {rnd.choice(LAST)}", org, role, resp, rnd.choice(WARDS),
                  rnd.choice(["Day", "Night", "On-call"]), f"x{rnd.randint(2000, 8999)}", rnd.choice(["Yes", "No"]),
                  rnd.randint(1, 30)])
save("emergency_staff.xlsx", ["staff_id", "full_name", "organisation", "role", "responsibility", "coverage_ward", "shift",
                              "phone_ext", "hazmat_certified", "years_experience"], staff)
