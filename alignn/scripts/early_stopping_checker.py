from jarvis.db.jsonutils import loadjson
import glob


def check_early_stoppping_reached(
    validation_file="history_val.json", n_early_stopping=30
):
    """Check if early stopping reached."""
    early_stopping_reached = False
    maes = loadjson(validation_file)["mae"]
    best_mae = 1e9
    no_improvement = 0
    for i in maes:
        if i > best_mae:
            no_improvement += 1
            if no_improvement == n_early_stopping:
                print("Reached Early Stopping at", i)
                early_stopping_reached = True
                best_mae = i
                break
        else:
            no_improvement = 0
            best_mae = i
    return early_stopping_reached, best_mae


def check_all_folders(path="."):
    for i in glob.glob(path + "/*/history_val.json"):
        print(i)
        early_stopping_reached, best_mae = check_early_stoppping_reached(
            validation_file=i
        )
        print(
            "early_stopping_reached,best_mae", early_stopping_reached, best_mae
        )
        print()


if __name__ == "__main__":
    # check_early_stoppping_reached()
    check_all_folders()
