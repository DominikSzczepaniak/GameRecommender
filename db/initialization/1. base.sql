CREATE TABLE Users (
    Id UUID PRIMARY KEY,
    Username VARCHAR(64) NOT NULL,
    Email VARCHAR(64) NOT NULL,
    Password VARCHAR(32) NOT NULL
);

CREATE TABLE UserToSteamId (
    UserId UUID PRIMARY KEY,
    SteamId VARCHAR(96) NOT NULL,
    CONSTRAINT fk_UserToSteamId_User FOREIGN KEY (UserId)
        REFERENCES Users(Id) ON DELETE CASCADE 
);

CREATE TABLE AppIdToName (
    AppId VARCHAR(64) PRIMARY KEY,
    Name VARCHAR(64) NOT NULL
);