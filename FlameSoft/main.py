from sys import exit, path

from openpyxl import load_workbook

path.append(r'E:\Github\Flame-Speed-Tool')
import FlameSoft.fs as fs


def string_to_list(var: str):
    """Convert the excel input to list"""
    try:
        ans = [int(num) for num in var.split(',')]
    except Exception as _:
        ans = []
    return ans


wb = load_workbook(r'E:\Github\Flame-Speed-Tool\FlameSoft\FlameSoft.xlsm', read_only=True, data_only=True)
ws = wb['Control_Sheet']

vals = dict(path=ws['F7'].value,
            slices=ws['F8'].value,
            filter=string_to_list(ws['F9'].value),
            thresh=string_to_list(ws['F10'].value),
            flow=ws['F11'].value,
            operation=ws['F12'].value,
            length=ws['F13'].value,
            fps=ws['F14'].value)

cls = fs.Flame(path=vals['path'])

try:
    if vals['operation'] == 1:
        points = fs.Crop(path=vals['path']).crop_video()
        cls.process(breaks=vals['slices'], filter_size=vals['filter'], thresh_val=vals['thresh'], crop_points=points,
                    flow_right=vals['flow'])
    elif vals['operation'] == 2:
        cls.whiten_image()
    elif vals['operation'] == 3:
        cls.blacken_image()
    elif vals['operation'] == 4:
        cls.get_data()
    elif vals['operation'] == 5:
        cls.view_pimage()
    elif vals['operation'] == 6:
        cls.get_data(length=vals['length'], fps=vals['fps'])
    exit()

except Exception as _:
    print(vals)
    exit()
