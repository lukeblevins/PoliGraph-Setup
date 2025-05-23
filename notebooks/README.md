# PoliGrapher Notebook

This folder contains a modified version of the existing notebook file that was available on Google Colab. It adds support for local execution in VSCode, pdf extraction, and bulk generation of knowledge graphs.

## Usage

1. **Install Python**: 
    
    This project requires a Python environment to be installed. Once Python is installed, proceed with the next steps.


2. **Open in Visual Studio Code (VSCode)**:

    Open the cloned poligraph-setup repository in VSCode:
    ```sh
    code ./
    ```

3. **Install Python Notebook Extension**:

    From the **Extensions** section on the left-hand side of VSCode, install the [Python Data Science extension pack](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.python-ds-extension-pack)

4. **Open the Notebook File**:

    Return to the **Explorer** section on the left-hand side of VSCode. Find the file project file `notebooks/poligrapher_notebook.ipynb` and open it.

5. **Edit the `policy_list.json` File**:

    Add the desired privacy policies to the json file. An example is provided below.

    ```json
    {
        "policy_urls": [
            {
                "name": "Target",
                "kind": "pdf",
                "path": "./policies/target.pdf"
            },
            {
                "name": "Ambetter Health",
                "kind": "webpage",
                "path": "https://www.ambetterhealth.com/en/privacy-policy/"
            }
        ]
    }
    ```

6. **Run the poligrapher notebook**:

    Return to the `poligrapher_notebook.ipynb` file. Click the **Run All** button found near the top. The notebook will install its dependencies, clone the latest version of the poligraph tool repository, and run it on the policies specified in the prior step.

> [!NOTE]
> The produced graphs and their visual representations will be stored in an `output/` directory. This folder is automatically created at the top level of this workspace.
