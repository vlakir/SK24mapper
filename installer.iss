#define MyAppName "SK42"
#define MyAppExeName "SK42.exe"
#define MyAppPublisher "SK42"
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
OutputBaseFilename=SK42-{#MyAppVersion}-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64os
; Путь относительно текущего .iss файла
SetupIconFile={#SourcePath}\img\icon.ico

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
; Копируем onedir-сборку PyInstaller в каталог установки
Source: "{#SourcePath}\dist\SK42\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

; Вариант A: копируем конфиги непосредственно в профиль пользователя
Source: "{#SourcePath}\configs\*"; DestDir: "{userappdata}\SK42\configs"; Flags: recursesubdirs ignoreversion

[Dirs]
; Создание обязательных директорий в профиле пользователя
Name: "{userappdata}\\SK42"
Name: "{userappdata}\\SK42\\configs"
Name: "{userappdata}\\SK42\\configs\\profiles"
Name: "{userappdata}\\SK42\\maps"
Name: "{localappdata}\\SK42"
Name: "{localappdata}\\SK42\\log"
Name: "{localappdata}\\SK42\\.cache\\tiles"

[InstallDelete]
; Удаляем старые пользовательские профили при обновлении
; (избегаем проблем совместимости со старыми версиями)
Type: files; Name: "{userappdata}\SK42\configs\profiles\*.toml"
; Удаляем старый ярлык SK42mapper на рабочем столе
Type: files; Name: "{userdesktop}\SK42mapper.lnk"

[UninstallDelete]
; Удаляем директории при деинсталляции, если пустые
Type: dirifempty; Name: "{userappdata}\\SK42\\configs\\profiles"
Type: dirifempty; Name: "{userappdata}\\SK42\\configs"
Type: dirifempty; Name: "{userappdata}\\SK42\\maps"
Type: dirifempty; Name: "{userappdata}\\SK42"
Type: dirifempty; Name: "{localappdata}\\SK42\\.cache\\tiles"
Type: dirifempty; Name: "{localappdata}\\SK42\\log"
Type: dirifempty; Name: "{localappdata}\\SK42"

[Icons]
; Ярлык на рабочем столе текущего пользователя (per-user установка)
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Запустить {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
var
  KeyChoicePage: TWizardPage;
  KeepKeyRadio: TNewRadioButton;
  ReplaceKeyRadio: TNewRadioButton;
  ExistingKeyFound: Boolean;
  KeyInputPage: TInputQueryWizardPage;

procedure MigrateFromOldName;
{ Переносит данные из SK42mapper в SK42 (одноразовая миграция при обновлении) }
var
  OldAppData, NewAppData, OldLocal, NewLocal: string;
  OldSecrets, NewSecrets: string;
  OldInstallDir: string;
  FindRec: TFindRec;
  ResultCode: Integer;
begin
  OldAppData := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42mapper';
  NewAppData := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42';
  OldLocal := AddBackslash(ExpandConstant('{localappdata}')) + 'SK42mapper';
  NewLocal := AddBackslash(ExpandConstant('{localappdata}')) + 'SK42';

  if not DirExists(OldAppData) then
    Exit;

  Log('Миграция: обнаружена старая папка ' + OldAppData);
  ForceDirectories(NewAppData);

  { 1. Перенос .secrets.env (API ключ) }
  OldSecrets := OldAppData + '\.secrets.env';
  NewSecrets := NewAppData + '\.secrets.env';
  if FileExists(OldSecrets) and not FileExists(NewSecrets) then
  begin
    Exec(ExpandConstant('{cmd}'), '/c attrib -R -H -S "' + OldSecrets + '"',
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    FileCopy(OldSecrets, NewSecrets, False);
    Exec(ExpandConstant('{cmd}'), '/c attrib +H "' + NewSecrets + '"',
         '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
    Log('Миграция: скопирован .secrets.env');
  end;

  { 2. Перенос профилей }
  ForceDirectories(NewAppData + '\configs\profiles');
  if FindFirst(OldAppData + '\configs\profiles\*.toml', FindRec) then
  begin
    try
      repeat
        if not FileExists(NewAppData + '\configs\profiles\' + FindRec.Name) then
        begin
          FileCopy(OldAppData + '\configs\profiles\' + FindRec.Name,
                   NewAppData + '\configs\profiles\' + FindRec.Name, False);
          Log('Миграция: скопирован профиль ' + FindRec.Name);
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;

  { 3. Перенос сохранённых карт }
  ForceDirectories(NewAppData + '\maps');
  if FindFirst(OldAppData + '\maps\*', FindRec) then
  begin
    try
      repeat
        if (FindRec.Name <> '.') and (FindRec.Name <> '..') then
        begin
          if not FileExists(NewAppData + '\maps\' + FindRec.Name) then
          begin
            FileCopy(OldAppData + '\maps\' + FindRec.Name,
                     NewAppData + '\maps\' + FindRec.Name, False);
            Log('Миграция: скопирована карта ' + FindRec.Name);
          end;
        end;
      until not FindNext(FindRec);
    finally
      FindClose(FindRec);
    end;
  end;

  { 4. Перенос кэша тайлов (только переименование папки, если новой нет) }
  if DirExists(OldLocal + '\.cache\tiles') and not DirExists(NewLocal + '\.cache\tiles') then
  begin
    ForceDirectories(NewLocal + '\.cache');
    RenameFile(OldLocal + '\.cache\tiles', NewLocal + '\.cache\tiles');
    Log('Миграция: перемещён кэш тайлов');
  end;

  { 5. Удаление старой папки установки }
  OldInstallDir := AddBackslash(ExpandConstant('{localappdata}')) + 'Programs\SK42mapper';
  if DirExists(OldInstallDir) then
  begin
    DelTree(OldInstallDir, True, True, True);
    Log('Миграция: удалена старая папка установки ' + OldInstallDir);
  end;
end;

procedure InitializeWizard;
var
  SecretsFile: string;
  InfoLabel: TNewStaticText;
begin
  { Миграция данных из SK42mapper → SK42 }
  MigrateFromOldName;

  { Проверяем наличие существующего ключа (уже после миграции) }
  SecretsFile := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42\.secrets.env';
  ExistingKeyFound := FileExists(SecretsFile);

  { Страница выбора: оставить или заменить ключ }
  KeyChoicePage := CreateCustomPage(
    wpSelectTasks,
    'API ключ',
    'Обнаружен сохранённый ключ API'
  );

  InfoLabel := TNewStaticText.Create(KeyChoicePage);
  InfoLabel.Parent := KeyChoicePage.Surface;
  InfoLabel.Caption := 'У вас уже есть сохранённый ключ API от предыдущей установки.'#13#10 +
                       'Выберите действие:';
  InfoLabel.Top := 0;
  InfoLabel.Left := 0;
  InfoLabel.AutoSize := True;

  KeepKeyRadio := TNewRadioButton.Create(KeyChoicePage);
  KeepKeyRadio.Parent := KeyChoicePage.Surface;
  KeepKeyRadio.Caption := 'Оставить текущий ключ (рекомендуется)';
  KeepKeyRadio.Top := InfoLabel.Top + InfoLabel.Height + 16;
  KeepKeyRadio.Left := 0;
  KeepKeyRadio.Width := KeyChoicePage.SurfaceWidth;
  KeepKeyRadio.Checked := True;

  ReplaceKeyRadio := TNewRadioButton.Create(KeyChoicePage);
  ReplaceKeyRadio.Parent := KeyChoicePage.Surface;
  ReplaceKeyRadio.Caption := 'Ввести новый ключ';
  ReplaceKeyRadio.Top := KeepKeyRadio.Top + KeepKeyRadio.Height + 8;
  ReplaceKeyRadio.Left := 0;
  ReplaceKeyRadio.Width := KeyChoicePage.SurfaceWidth;

  { Страница ввода ключа }
  KeyInputPage := CreateInputQueryPage(
    KeyChoicePage.ID,
    'Конфигурация ключа',
    'Введите ключ API',
    'Ключ будет сохранён в профиле пользователя (%AppData%\SK42).'
  );
  KeyInputPage.Add('API_KEY:', True);
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;
  { Пропускаем страницу выбора, если ключа ещё нет }
  if PageID = KeyChoicePage.ID then
    Result := not ExistingKeyFound;
  { Пропускаем страницу ввода, если пользователь решил оставить текущий ключ }
  if PageID = KeyInputPage.ID then
    Result := ExistingKeyFound and KeepKeyRadio.Checked;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = KeyInputPage.ID then
  begin
    if Trim(KeyInputPage.Values[0]) = '' then
    begin
      MsgBox('Введите ключ API.', mbError, MB_OK);
      Result := False;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  BaseDir, F, Content: string;
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    BaseDir := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42';
    ForceDirectories(BaseDir);
    F := BaseDir + '\.secrets.env';

    { Если пользователь выбрал "оставить текущий" — ничего не делаем }
    if ExistingKeyFound and KeepKeyRadio.Checked then
    begin
      Log('Сохраняем существующий .secrets.env без изменений: ' + F);
      Exit;
    end;

    Content := 'API_KEY=' + Trim(KeyInputPage.Values[0]) + #13#10;

    { Снимаем атрибуты со старого файла, если он есть }
    if FileExists(F) then
    begin
      if not Exec(ExpandConstant('{cmd}'), '/c attrib -R -H -S "' + F + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        Log('Не удалось снять атрибуты (attrib -R -H -S). Код=' + IntToStr(ResultCode));
      if not DeleteFile(F) then
        Log('Предупреждение: не удалось удалить старый файл перед записью: ' + F);
    end;

    { Записываем новый файл }
    if not SaveStringToFile(F, Content, False) then
    begin
      MsgBox('Не удалось записать файл: ' + F + #13#10 +
             'Попробуйте закрыть приложение и переустановить, либо запустите установщик от имени текущего пользователя.',
             mbError, MB_OK);
    end
    else
    begin
      { Скрываем файл через системную утилиту attrib }
      if not Exec(ExpandConstant('{cmd}'), '/c attrib +H "' + F + '"', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
        Log(Format('Не удалось скрыть файл атрибутом Hidden, код=%d', [ResultCode]));
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  F: string;
begin
  if CurUninstallStep = usUninstall then
  begin
    F := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42' + '\.secrets.env';
    DeleteFile(F);
  end;
end;
