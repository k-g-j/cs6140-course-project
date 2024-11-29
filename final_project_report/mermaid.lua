-- mermaid.lua
local pandoc = require("pandoc")

-- Function to escape special characters in filenames
local function sanitize_filename(name)
  local s = name:gsub("%s+", "_")
  s = s:gsub("[^A-Za-z0-9_%-%.]", "")
  return s
end

function CodeBlock(elem)
  if elem.classes:includes("mermaid") then
    -- Generate a unique filename based on current time and a counter
    local filename = "mermaid_diagram_" .. os.time() .. "_" .. math.random(1000,9999) .. ".png"
    
    -- Write the Mermaid code to a temporary .mmd file
    local temp_mmd = "temp_" .. filename .. ".mmd"
    local file = io.open(temp_mmd, "w")
    file:write(elem.text)
    file:close()
    
    -- Call Mermaid CLI to convert the .mmd file to a PNG image
    local command = string.format("mmdc -i %s -o %s", temp_mmd, filename)
    os.execute(command)
    
    -- Remove the temporary .mmd file
    os.remove(temp_mmd)
    
    -- Return an Image element to replace the Mermaid code block
    return pandoc.Para({pandoc.Image("Mermaid Diagram", filename)})
  end
end