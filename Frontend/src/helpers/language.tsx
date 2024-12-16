function getLanguage() {
  return localStorage.getItem("language") || "en";
}

export function getLanguageFile(){
    switch(getLanguage()){
        case "pl":
            return require("../translations/pl.json");
        default:
            return require("../translations/en.json");
    }
} 