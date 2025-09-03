#define MyAppName "SK42mapper"
#define MyAppExeName "SK42mapper.exe"
#define MyAppPublisher "SK42mapper"
#define MyAppVersion GetEnv("VERSION")

[Setup]
AppId={{72A4E6B5-5F2F-4F1D-9A44-51FA6B5CFE01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Install per-user without admin rights
DefaultDirName={localappdata}\Programs\{#MyAppName}
PrivilegesRequired=lowest
; Do not reuse previous install dir (avoid Program Files if previously installed)
UsePreviousAppDir=no
; Keep directory page but with correct default
DisableDirPage=auto
OutputDir=.
OutputBaseFilename=SK42mapper-{#MyAppVersion}-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64os
; Путь относительно текущего .iss файла (scripts)
SetupIconFile={#SourcePath}\..\img\icon.ico

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
; Копируем onedir-сборку PyInstaller в каталог установки
Source: "{#SourcePath}\..\dist\SK42mapper\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Dirs]
; Создание обязательных директорий в профиле пользователя
Name: "{userappdata}\\SK42mapper"
Name: "{userappdata}\\SK42mapper\\configs"
Name: "{userappdata}\\SK42mapper\\configs\\profiles"
Name: "{userappdata}\\SK42mapper\\maps"
Name: "{localappdata}\\SK42mapper"
Name: "{localappdata}\\SK42mapper\\log"
Name: "{localappdata}\\SK42mapper\\.cache\\tiles"

[UninstallDelete]
; Удаляем директории при деинсталляции, если пустые
Type: dirifempty; Name: "{userappdata}\\SK42mapper\\configs\\profiles"
Type: dirifempty; Name: "{userappdata}\\SK42mapper\\configs"
Type: dirifempty; Name: "{userappdata}\\SK42mapper\\maps"
Type: dirifempty; Name: "{userappdata}\\SK42mapper"
Type: dirifempty; Name: "{localappdata}\\SK42mapper\\.cache\\tiles"
Type: dirifempty; Name: "{localappdata}\\SK42mapper\\log"
Type: dirifempty; Name: "{localappdata}\\SK42mapper"

[Icons]
; Рабочая директория ярлыка — {app}, чтобы относительные пути в приложении были корректны
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
var
  SecretsPage: TInputQueryWizardPage;

procedure InitializeWizard;
begin
  SecretsPage := CreateInputQueryPage(
    wpSelectTasks,
    'Конфигурация ключа',
    'Введите ключ API',
    'Ключ будет сохранён в профиле пользователя (%AppData%\\SK42mapper).'
  );
  { Второй параметр True — скрытый ввод, как пароль }
  SecretsPage.Add('API_KEY:', True);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = SecretsPage.ID then
  begin
    if Trim(SecretsPage.Values[0]) = '' then
    begin
      MsgBox('Введите ключ API.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  F, Content: string;
  ResultCode: Integer;
begin
  if CurStep = ssInstall then
  begin
    F := ExpandConstant('{userappdata}\\SK42mapper\\.secrets.env');
    Content := 'API_KEY=' + SecretsPage.Values[0] + #13#10;
    if not SaveStringToFile(F, Content, False) then
    begin
      MsgBox('Не удалось записать файл: ' + F, mbError, MB_OK);
    end
    else
    begin
      { Скрываем файл через системную утилиту attrib }
      if not Exec(ExpandConstant('{cmd}'), '/c attrib +H "' + F + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      begin
        Log(Format('Не удалось скрыть файл атрибутом Hidden, код=%d', [ResultCode]));
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    DeleteFile(ExpandConstant('{userappdata}\\SK42mapper\\.secrets.env'));
  end;
end;
