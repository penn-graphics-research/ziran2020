import subprocess

commands = []
commands.append("./coupling -test 1001 | grep \"frame\"")
commands.append("./coupling -test 1002 | grep \"frame\"")
commands.append("./coupling -test 1003 | grep \"frame\"")
commands.append("./coupling -test 1004 | grep \"frame\"")
commands.append("./coupling -test 1005 | grep \"frame\"")
commands.append("./coupling -test 1006 | grep \"frame\"")
commands.append("./coupling -test 1007 | grep \"frame\"")
commands.append("./coupling -test 1008 | grep \"frame\"")
commands.append("./coupling -test 1009 | grep \"frame\"")

subprocess.call(["rm -rf output/standardtest_*"], shell=True)
for command in commands:
    subprocess.call([command], shell=True)
