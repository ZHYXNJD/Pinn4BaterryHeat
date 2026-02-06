import numpy as np
import pandas as pd
from torch.utils.data import dataset


def process_dt():

    # 文件路径
    temp_file = "1c40采样点数据.xlsx"
    coord_file = "sample_data_coords.xlsx"
    test_data_point = pd.read_excel('test_data_point.xlsx')
    test_data_name = test_data_point['name'].values.tolist()

    # 初始化结果列表
    train_interior_dataset = []

    test_dataset = []

    # 遍历 17 个 sheet
    for i in range(1, 18):
        # 读取温度数据
        temp_df = pd.read_excel(temp_file, sheet_name=i - 1)  # sheet index 从0开始
        # if i != 5:
        #     pass
        # else:
        #     time_col = temp_df.columns[0]
        #     temp_df = temp_df[temp_df[time_col] % 1 == 0].reset_index(drop=True)
        # 读取坐标数据
        coord_df = pd.read_excel(coord_file, sheet_name=i - 1)

        coord_df = coord_df.iloc[:, 1:]
        coord_df['x'] = coord_df['x'] - 264
        coord_df['y'] = coord_df['y'] - 70.5
        coord_df['z'] = coord_df['z'] - 109

        # 建立坐标字典 {name: (x,y,z)}
        coord_map = {row['name']: (row['x'], row['y'], row['z']) for _, row in coord_df.iterrows()}

        # 遍历每个采样点列（除第一列时间）
        for col in temp_df.columns[1:]:
            # 提取采样点名称，例如 te1(a1) → A1
            point_name = col.split("(")[-1].strip(")").upper()
            data_point_index = point_name[1:]
            x, y, z = coord_map[point_name]

            # 遍历时间和温度
            for t, temp in zip(temp_df.iloc[:, 0], temp_df[col]):
                if col not in test_data_name:
                    # if data_point_index not in boundary_data_point_number:
                    #     train_interior_dataset.append([x, y, z, t, temp])
                    # else:
                    #     train_boundary_dataset.append([x, y, z, t, temp])
                    train_interior_dataset.append([x, y, z, t, temp])
                else:
                    test_dataset.append([x, y, z, t, temp])

    # 转换为 DataFrame
    final_train_interior_df = pd.DataFrame(train_interior_dataset, columns=["x", "y", "z", "t", "temperature"])
    # final_train_boundary_df = pd.DataFrame(train_boundary_dataset, columns=["x", "y", "z", "t", "temperature"])
    final_test_df = pd.DataFrame(test_dataset, columns=["x", "y", "z", "t", "temperature"])

    # 保存为 CSV
    final_train_interior_df.to_csv(f"battery_train_dataset_1C40.csv", index=False)
    # final_train_boundary_df.to_csv("battery_train_boundary_dataset.csv", index=False)
    final_test_df.to_csv("battery_test_dataset_1C40.csv", index=False)

    # np.save("battery_train_interior_dataset.npy", final_train_interior_df)
    # np.save("battery_train_dataset.npy", final_train_interior_df)
    # np.save("battery_train_boundary_dataset.npy", final_train_boundary_df)
    # np.save("battery_test_dataset.npy", final_test_df)

if __name__ == "__main__":
    process_dt()
