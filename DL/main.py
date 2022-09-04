from video_process import process_video
from data_process import process_data, radar_all, new_process_data

if __name__ == '__main__':
    video_path = ""
    video_path = "dataset/"
    result_path = "result/"
    full_result = "full_result.csv"
    result = "result_clear.csv"
    process_video(video_path, result_path + full_result)
    new_process_data(result_path + full_result, result_path + result)
    #radar_all(result_path, "result")