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
; Путь относительно текущего .iss файла
SetupIconFile={#SourcePath}\img\icon.ico

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Files]
; Копируем onedir-сборку PyInstaller в каталог установки
Source: "{#SourcePath}\dist\SK42mapper\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

; Вариант A: копируем конфиги непосредственно в профиль пользователя
Source: "{#SourcePath}\configs\*"; DestDir: "{userappdata}\SK42mapper\configs"; Flags: recursesubdirs ignoreversion

[Dirs]
; Создание обязательных директорий в профиле пользователя
Name: "{userappdata}\\SK42mapper"
Name: "{userappdata}\\SK42mapper\\configs"
Name: "{userappdata}\\SK42mapper\\configs\\profiles"
Name: "{userappdata}\\SK42mapper\\maps"
Name: "{localappdata}\\SK42mapper"
Name: "{localappdata}\\SK42mapper\\log"
Name: "{localappdata}\\SK42mapper\\.cache\\tiles"

[InstallDelete]
; Удаляем старые пользовательские профили при обновлении
; (избегаем проблем совместимости со старыми версиями)
Type: files; Name: "{userappdata}\SK42mapper\configs\profiles\*.toml"

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

procedure InitializeWizard;
var
  SecretsFile: string;
  InfoLabel: TNewStaticText;
begin
  { Проверяем наличие существующего ключа }
  SecretsFile := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42mapper\.secrets.env';
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
    'Ключ будет сохранён в профиле пользователя (%AppData%\SK42mapper).'
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
    BaseDir := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42mapper';
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
    F := AddBackslash(ExpandConstant('{userappdata}')) + 'SK42mapper' + '\.secrets.env';
    DeleteFile(F);
  end;
end;
