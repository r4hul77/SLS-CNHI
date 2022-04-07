from utils.filehandling import *
import openpyxl


def append(org_datasheet, to_beappended):
    sheet = to_beappended.worksheets[0]
    for i, row in enumerate(sheet.rows):
        if i== 0:
            continue
        org_datasheet.worksheets[0].append([cell.value for cell in row])
    return org_datasheet


if __name__ == "__main__":
    path = "/home/harsha/Desktop/SLS-CNH/New folder"
    ret = LoadFilesWithExtensions(path, ["xlsx"])
    org = openpyxl.load_workbook(ret[0])
    for workbook in ret[1:]:
        org = append(org, openpyxl.load_workbook(workbook))
    org.save(path+"RESULT.xlsx")
