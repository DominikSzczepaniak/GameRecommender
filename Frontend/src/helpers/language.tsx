function getLanguage() {
  return localStorage.getItem("language") || "en";
}

export function getLanguageFile(){
    switch(getLanguage()){
        case "pl":
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            return require("../translations/pl.json");
        default:
            // eslint-disable-next-line @typescript-eslint/no-require-imports
            return require("../translations/en.json");
    }
} 