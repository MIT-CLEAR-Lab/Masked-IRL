import os

# s = """
#   - features: ["table", "laptop", "proxemics", "coffee"] #, "proxemics", "human", "coffee"]
#     feature_scaling: "normalize"
#
#     # Demonstration generation preferences for reward.
#     preferencer:
#       theta: {}
#       beta: 20.0
#       f_method: "boltzmann"
#       s_method: "luce"
#
# """

s = """
  - features: [ "computer_dist", "joint_up", "feat_obst1", "feat_obst2" ]
    feature_scaling: "normalize"

    # Demonstration generation preferences for reward.
    preferencer:
      theta: {}
      beta: 10.0
      f_method: "boltzmann"
      s_method: "luce"
"""

with open(os.path.join("gridrobot4.yaml"), 'w') as f:
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                for d in [-1, 0, 1]:
                    f.write(s.replace("{}", "[{}, {}, {}, {}]".format(a, b, c, d)))