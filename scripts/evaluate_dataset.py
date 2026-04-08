from run_slam import slam

def main():

    room_names = ["20", "21", "22", "23", "24"]

    for room_name in room_names:
        slam(room_name)

if __name__ == "__main__":
    main()