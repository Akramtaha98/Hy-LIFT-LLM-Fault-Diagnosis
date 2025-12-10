# Name : Abd Al Hameed Mohammed Younis 
# deparment: network / morning study 

while True:
    text = input("Enter numbers separated by spaces (or 'exit' to quit): ")

    if text.lower() == "exit":
        print("Exiting program...")
        break

    numbers = text.split()
    all_good = True

    for n in numbers:
        if not n.isdigit() and not (n.startswith('-') and n[1:].isdigit()):
            print("Invalid input.")
            all_good = False
            break

    if not all_good:
        continue

    for i in [int(n) for n in numbers]:
        if i < 0:
            print(f"Negative skipped: {i}")
            continue
        if i == 0:
            print("Zero found. Stopping early.")
            break
        if i % 2 == 0:
            print(f"Even number: {i}")
            if i % 4 == 0:
                print("Also divisible by 4")
        else:
            print(f"Odd number: {i}")
    else:
        print("All numbers processed successfully.")
