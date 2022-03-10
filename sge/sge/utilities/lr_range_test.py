from examples.model_evaluator import fake_train_model, train_model
from bee_bot.flower import create_report
import time
def learning_rate_range_test(minimum, maximum, number):
    record = []
    start_time = time.time()
    for i in range(number):
        learning_rate = minimum + (maximum - minimum) / number * i
        #results = fake_train_model(learning_rate)
        results = train_model(str(learning_rate))
        record.append([learning_rate, results])
        progress_percentage = float(i + 1) / number
        elapsed_time = time.time() - start_time
        time_estimate = elapsed_time / progress_percentage
        print(time_estimate)
        hours_left = int(time_estimate / 60 / 60)
        minutes_left = int(time_estimate % 60)
        seconds_left = int(time_estimate / 60 % 60)
        print(hours_left, minutes_left, seconds_left)
        print(learning_rate)
        create_report('lr_range_test_report.json', 'bee_reports/',
            {'Percentage:': progress_percentage,
            'Time left:': str(hours_left) + ':' + str(minutes_left) + ':' + str(seconds_left)
            },record)
    return record


def visualize_test_record(record):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    lr_list = [x[0] for x in record]
    result_list = [x[1][0] for x in record]
    plt.plot(lr_list, result_list)
    plt.savefig("admissiblelr.pdf")

if __name__ == "__main__":
    record = learning_rate_range_test(0,1,500)
    visualize_test_record(record)

