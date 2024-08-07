/* eslint-disable react/prop-types */
import { Popover } from "react-tiny-popover";
import { useSelector, useDispatch } from "react-redux";
import { IoIosArrowDown } from "react-icons/io";
import { FiFramer } from "react-icons/fi";
import { PiTextT } from "react-icons/pi";
import styles from "./ModelSelect.module.css";
import { getVideoAction } from "../../Redux/Actions/QueryVideoActions";

const ModelSelect = (props) => {
  const dispatch = useDispatch();
  const { isOpen, setOpenSelect } = props;
  const { query } = useSelector((state) => state.queryVideoSlice);
  const handleChangeMode = (value) => {
    dispatch(
      getVideoAction({
        query: query,
        mode: value,
      })
    );
    setOpenSelect(false);
  }
  return (
    <>
      <Popover
        isOpen={isOpen}
        positions={["top", "bottom", "left", "right"]}
        content={
          <div className={styles["listModel"]}>
            <div className={styles.listModelContainer}>
              <div
                className={styles.modelItem}
                onClick={() => handleChangeMode("keyframe")}
              >
                <div className={styles.divIcon}>
                  <FiFramer />
                </div>
                <p>Keyframe</p>
              </div>

              <div
                className={styles.modelItem}
                onClick={() => handleChangeMode("text")}
              >
                <div className={styles.divIcon}>
                  <PiTextT />
                </div>
                <p> Text features</p>
              </div>
            </div>
          </div>
        }
        onClickOutside={() => setOpenSelect(false)}
      >
        <div className={styles.selectModel}>
          <div
            className={
              styles["selectModelContainer"] +
              " " +
              styles[isOpen ? "showGrayBg" : ""]
            }
            onClick={() => setOpenSelect(true)}
          >
            <p>ChatGPT</p>
            <IoIosArrowDown />
          </div>
        </div>
      </Popover>
    </>
  );
};

export default ModelSelect;