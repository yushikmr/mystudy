Attribute VB_Name = "Module2"
Sub OutputJitterWithDynamicPositionAndAutoUpdate()
    ' ======== ? �ݒ� ========
    Const sourceSheetName As String = "Sheet2"
    Const outputSheetName As String = "Output"
    Const valueColumnHeader As String = "Value"
    Const binCellAddress As String = "K1"
    Const aCellAddress As String = "K2"
    Const bCellAddress As String = "K3"
    ' ==========================

    Dim srcWs As Worksheet, outWs As Worksheet
    Set srcWs = Worksheets(sourceSheetName)

    ' �o�̓V�[�g�����i�Ȃ���΍쐬�j
    On Error Resume Next
    Set outWs = Worksheets(outputSheetName)
    If outWs Is Nothing Then
        Set outWs = Worksheets.Add(After:=srcWs)
        outWs.Name = outputSheetName
    End If
    On Error GoTo 0

    ' �ݒ�l�ǂݍ���
    Dim binWidth As Double, aVal As Double, bVal As Double
    binWidth = srcWs.Range(binCellAddress).Value
    aVal = srcWs.Range(aCellAddress).Value
    bVal = srcWs.Range(bCellAddress).Value

    ' �Ώۗ�̎擾
    Dim valueCol As Long
    valueCol = 0
    Dim colIdx As Long
    For colIdx = 1 To srcWs.Cells(1, Columns.count).End(xlToLeft).Column
        If srcWs.Cells(1, colIdx).Value = valueColumnHeader Then
            valueCol = colIdx
            Exit For
        End If
    Next colIdx
    If valueCol = 0 Then
        MsgBox "�Ώۗ񂪌�����܂���", vbCritical
        Exit Sub
    End If

    ' �f�[�^�͈�
    Dim lastRow As Long
    lastRow = srcWs.Cells(srcWs.Rows.count, "A").End(xlUp).Row

    ' ==== ?? Output�ʒu�̎������� ====
    Dim startCol As Long
    startCol = outWs.Cells(1, outWs.Columns.count).End(xlToLeft).Column
    If outWs.Cells(1, startCol).Value <> "" Then
        startCol = startCol + 1
    End If
    ' �o�͗�F�W�b�^�[ / �l / Bin
    Dim jitterCol As Long: jitterCol = startCol
    Dim valueColOut As Long: valueColOut = startCol + 1
    Dim binColOut As Long: binColOut = startCol + 2

    ' ==== ??? �w�b�_�[�ݒ� ====
    outWs.Cells(1, jitterCol).Value = "�W�b�^�[_" & valueColumnHeader
    outWs.Cells(1, valueColOut).Value = valueColumnHeader
    outWs.Cells(1, binColOut).Value = valueColumnHeader & "_Bin"

    ' ==== ?? �f�[�^���� ====
    Dim i As Long, j As Long
    For i = 2 To lastRow
        Dim dateVal As Variant, val As Variant, binVal As Double
        dateVal = srcWs.Cells(i, 1).Value
        val = srcWs.Cells(i, valueCol).Value
        binVal = WorksheetFunction.Floor(val, binWidth)

        ' �o�����J�E���g�i�s���[�v�ň���J�E���g�j
        Dim count As Long: count = 1
        For j = 2 To i - 1
            If srcWs.Cells(j, 1).Value = dateVal And _
               WorksheetFunction.Floor(srcWs.Cells(j, valueCol).Value, binWidth) = binVal Then
                count = count + 1
            End If
        Next j

        ' �W�b�^�[�v�Z
        Dim jitterOffset As Double
        jitterOffset = WorksheetFunction.Min(aVal * (count - 1), 0.5)
        If Rnd() < 0.5 Then jitterOffset = -jitterOffset
        Dim jitterVal As Double
        jitterVal = dateVal + jitterOffset + bVal

        ' �o��
        outWs.Cells(i, jitterCol).Value = jitterVal
        outWs.Cells(i, valueColOut).Value = val
        outWs.Cells(i, binColOut).Value = binVal
    Next i

    MsgBox "�o�̓V�[�g�ɒǋL���܂����i������͕ێ��j", vbInformation
End Sub


